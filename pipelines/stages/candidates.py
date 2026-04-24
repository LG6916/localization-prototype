"""Candidate generation: DBSCAN clustering, FPFH+RANSAC, sphere RANSAC."""
from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from ..base import Stage, StageResult, PipelineContext, Detection


def _to_open3d(points, colors=None, normals=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


# --- Sphere fit via algebraic + least-squares -------------------------------

def _fit_sphere_ls(pts: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Algebraic sphere fit. Returns (center, radius, rmse)."""
    if len(pts) < 4:
        return np.zeros(3), 0.0, float("inf")
    A = np.hstack([2 * pts, np.ones((len(pts), 1))])
    b = np.sum(pts ** 2, axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r2 = sol[3] + np.dot(c, c)
    if r2 <= 0:
        return c, 0.0, float("inf")
    r = float(np.sqrt(r2))
    d = np.linalg.norm(pts - c, axis=1) - r
    rmse = float(np.sqrt(np.mean(d * d)))
    return c, r, rmse


def _pca_align(roi_pts: np.ndarray, model_pts: np.ndarray) -> tuple[np.ndarray, float]:
    """PCA-based initial alignment: match principal axes of `model_pts` to
    `roi_pts`. Returns (4x4 T mapping model→scene, mean chamfer distance).

    Tries the 4 det-preserving flip ambiguities of PCA and keeps the best by
    mean point-to-point distance. Sphere-like inputs return a valid (but
    meaningless) rotation — callers should have handled that case already.
    """
    if len(roi_pts) < 4 or len(model_pts) < 4:
        T = np.eye(4)
        T[:3, 3] = roi_pts.mean(0) if len(roi_pts) else 0.0
        return T, float("inf")
    c_roi = roi_pts.mean(0)
    c_model = model_pts.mean(0)
    P_r = roi_pts - c_roi
    P_m = model_pts - c_model
    cov_r = (P_r.T @ P_r) / max(len(P_r), 1)
    cov_m = (P_m.T @ P_m) / max(len(P_m), 1)
    ev_r, V_r = np.linalg.eigh(cov_r)
    ev_m, V_m = np.linalg.eigh(cov_m)
    # eigh returns ascending; reverse for descending-eigenvalue order
    V_r = np.fliplr(V_r)
    V_m = np.fliplr(V_m)
    if np.linalg.det(V_r) < 0:
        V_r[:, 2] *= -1
    if np.linalg.det(V_m) < 0:
        V_m[:, 2] *= -1

    tree = cKDTree(roi_pts)
    flips = [(1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)]
    best_T = None
    best_cost = float("inf")
    for fx, fy, fz in flips:
        F = np.diag([fx, fy, fz]).astype(np.float64)
        R = V_r @ F @ V_m.T
        if np.linalg.det(R) < 0:
            continue
        t = c_roi - R @ c_model
        transformed = (R @ model_pts.T).T + t
        d, _ = tree.query(transformed, k=1)
        cost = float(d.mean())
        if cost < best_cost:
            best_cost = cost
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            best_T = T
    if best_T is None:
        best_T = np.eye(4)
        best_T[:3, 3] = c_roi
    return best_T, best_cost


def _ransac_sphere(
    pts: np.ndarray,
    radius_prior_mm: float | None,
    inlier_tol_mm: float,
    iterations: int = 200,
    radius_tolerance_frac: float = 0.15,
    min_inliers: int = 40,
    seed: int = 0,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """RANSAC sphere fit. Returns (center, radius, inlier_mask) or None."""
    rng = np.random.default_rng(seed)
    best = None
    best_score = 0
    n = len(pts)
    if n < 4:
        return None
    r_lo = radius_prior_mm * (1 - radius_tolerance_frac) if radius_prior_mm else None
    r_hi = radius_prior_mm * (1 + radius_tolerance_frac) if radius_prior_mm else None
    for _ in range(iterations):
        idx = rng.choice(n, size=4, replace=False)
        c, r, _ = _fit_sphere_ls(pts[idx])
        if not np.isfinite(r) or r <= 0:
            continue
        if r_lo is not None and (r < r_lo or r > r_hi):
            continue
        d = np.abs(np.linalg.norm(pts - c, axis=1) - r)
        inliers = d < inlier_tol_mm
        score = int(inliers.sum())
        if score > best_score:
            # refine with inliers
            c2, r2, _ = _fit_sphere_ls(pts[inliers])
            if np.isfinite(r2) and r2 > 0:
                if r_lo is None or (r_lo <= r2 <= r_hi):
                    best = (c2, r2, inliers)
                    best_score = score
    if best is None or best_score < min_inliers:
        return None
    return best


class CandidatesStage(Stage):
    slot = "candidates"

    METHODS = ["dbscan", "sphere_ransac_per_cluster", "feature_ransac",
               "cluster_centers", "twod_rois", "twod_feature_rois"]

    def __init__(self, method: str = "dbscan", params=None):
        defaults = dict(
            # DBSCAN
            dbscan_eps_mm=6.0,
            dbscan_min_points=25,
            # sphere ransac
            radius_tolerance_frac=0.15,
            sphere_inlier_tol_mm=1.5,
            sphere_ransac_iters=400,
            sphere_min_inliers=60,
            # feature ransac
            voxel_model_mm=2.0,
            voxel_scene_mm=3.0,
            feature_radius_mul=5.0,
            ransac_n=3,
            ransac_iters=100000,
            ransac_confidence=0.999,
            mutual_filter=True,
            max_candidates=40,
            # twod_rois
            roi_min_points=30,
            roi_sphere_inlier_tol_mm=1.5,
            roi_sphere_ransac_iters=200,
            roi_bbox_dilate_px=4,
        )
        defaults.update(params or {})
        super().__init__(defaults)
        self.method = method

    # --- dbscan ---
    def _dbscan(self, ctx: PipelineContext) -> dict:
        pts = ctx.current_points
        pcd = _to_open3d(pts)
        labels = np.array(
            pcd.cluster_dbscan(
                eps=float(self.params["dbscan_eps_mm"]),
                min_points=int(self.params["dbscan_min_points"]),
                print_progress=False,
            )
        )
        ctx.cluster_labels = labels
        # create coarse "cluster center" detections with identity rotation
        dets: list[Detection] = []
        uniq = sorted(int(l) for l in set(labels) if l >= 0)
        for i, lbl in enumerate(uniq):
            m = labels == lbl
            if m.sum() < 4:
                continue
            c = pts[m].mean(axis=0)
            pose = np.eye(4)
            pose[:3, 3] = c
            d = Detection(
                instance_id=i,
                pose=pose,
                confidence=float(m.sum()) / len(pts),
                method="cluster_center",
                extra={"cluster_id": int(lbl), "n_points": int(m.sum())},
            )
            dets.append(d)
        ctx.candidates = dets
        return {"n_clusters": len(uniq), "noise_fraction": float((labels < 0).mean())}

    # --- per-cluster sphere ransac ---
    def _sphere_per_cluster(self, ctx: PipelineContext) -> dict:
        pts = ctx.current_points
        # ensure we have clusters; if not, quick dbscan
        if ctx.cluster_labels is None or len(ctx.cluster_labels) != len(pts):
            pcd = _to_open3d(pts)
            ctx.cluster_labels = np.array(
                pcd.cluster_dbscan(
                    eps=float(self.params["dbscan_eps_mm"]),
                    min_points=int(self.params["dbscan_min_points"]),
                    print_progress=False,
                )
            )
        labels = ctx.cluster_labels
        uniq = sorted(int(l) for l in set(labels) if l >= 0)
        dets: list[Detection] = []
        for i, lbl in enumerate(uniq):
            m = labels == lbl
            sub = pts[m]
            res = _ransac_sphere(
                sub,
                radius_prior_mm=ctx.model_radius_mm,
                inlier_tol_mm=float(self.params["sphere_inlier_tol_mm"]),
                iterations=int(self.params["sphere_ransac_iters"]),
                radius_tolerance_frac=float(self.params["radius_tolerance_frac"]),
                min_inliers=int(self.params["sphere_min_inliers"]),
                seed=1000 + i,
            )
            if res is None:
                continue
            c, r, inl = res
            pose = np.eye(4)
            pose[:3, 3] = c
            conf = float(inl.sum()) / max(len(sub), 1)
            fitness = float(inl.sum()) / max(len(sub), 1)
            err_r = abs(r - (ctx.model_radius_mm or r))
            d = Detection(
                instance_id=len(dets),
                pose=pose,
                confidence=conf,
                fitness=fitness,
                n_inliers=int(inl.sum()),
                method="sphere_ransac",
                extra={
                    "fitted_radius_mm": float(r),
                    "radius_error_mm": float(err_r),
                    "cluster_id": int(lbl),
                    "cluster_size": int(m.sum()),
                },
            )
            dets.append(d)
            if ctx.progress:
                ctx.progress.emit((i + 1) / max(len(uniq), 1), f"sphere fit {i+1}/{len(uniq)}")
        ctx.candidates = dets
        return {"n_clusters": len(uniq), "n_spheres": len(dets)}

    # --- cluster centers as coarse seeds (used by cluster-seeded ICP) ---
    def _cluster_centers(self, ctx: PipelineContext) -> dict:
        """DBSCAN clusters, each → one candidate seed.

        For sphere-like models we just use the cluster centroid with identity
        rotation (sphere rotation is ambiguous anyway). For any other model we
        PCA-align the model's principal axes to the cluster's axes so ICP gets
        a rotation seed that's at most one flip ambiguity away from truth —
        dramatically improves convergence on elongated parts like bolts.
        """
        pts = ctx.current_points
        pcd = _to_open3d(pts)
        labels = np.array(
            pcd.cluster_dbscan(
                eps=float(self.params["dbscan_eps_mm"]),
                min_points=int(self.params["dbscan_min_points"]),
                print_progress=False,
            )
        )
        ctx.cluster_labels = labels
        uniq = sorted(int(l) for l in set(labels) if l >= 0)
        is_sphere = ctx.model_radius_mm is not None
        dets: list[Detection] = []
        for i, lbl in enumerate(uniq):
            m = labels == lbl
            if m.sum() < 4:
                continue
            cluster_pts = pts[m].astype(np.float32)
            if is_sphere:
                pose = np.eye(4)
                pose[:3, 3] = cluster_pts.mean(axis=0)
                extra = {"cluster_id": int(lbl), "n_points": int(m.sum()),
                          "seed_kind": "centroid"}
            else:
                pose, pca_cost = _pca_align(cluster_pts, ctx.model_points)
                extra = {"cluster_id": int(lbl), "n_points": int(m.sum()),
                          "seed_kind": "pca_align",
                          "pca_chamfer_mean_mm": float(pca_cost)}
            dets.append(Detection(
                instance_id=i,
                pose=pose,
                confidence=float(m.sum()) / len(pts),
                method="cluster_center",
                extra=extra,
            ))
        ctx.candidates = dets
        return {
            "n_clusters": len(uniq),
            "noise_fraction": float((labels < 0).mean()),
            "seed_kind": "pca_align" if not is_sphere else "centroid",
        }

    # --- feature-based global registration ---
    def _feature_ransac(self, ctx: PipelineContext) -> dict:
        vs = float(self.params["voxel_scene_mm"])
        vm = float(self.params["voxel_model_mm"])
        scene_pcd = _to_open3d(ctx.current_points, ctx.current_colors, ctx.current_normals)
        scene_ds = scene_pcd.voxel_down_sample(vs)
        if not scene_ds.has_normals():
            scene_ds.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=vs * 2.0, max_nn=30)
            )
        model_pcd = o3d.geometry.PointCloud()
        model_pcd.points = o3d.utility.Vector3dVector(ctx.model_points)
        model_ds = model_pcd.voxel_down_sample(vm)
        model_ds.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=vm * 2.0, max_nn=30)
        )
        fr = float(self.params["feature_radius_mul"]) * max(vs, vm)
        fpfh_s = o3d.pipelines.registration.compute_fpfh_feature(
            scene_ds, o3d.geometry.KDTreeSearchParamHybrid(radius=fr, max_nn=100)
        )
        fpfh_m = o3d.pipelines.registration.compute_fpfh_feature(
            model_ds, o3d.geometry.KDTreeSearchParamHybrid(radius=fr, max_nn=100)
        )
        if ctx.progress: ctx.progress.emit(0.4, "FPFH computed")

        # We iterate a few times to collect multiple candidates. Open3D's RANSAC
        # returns a single best transform; to get multiple we randomize input
        # subsets of the scene by masking already-explained regions.
        candidates: list[Detection] = []
        scene_pts_np = np.asarray(scene_ds.points)
        explained_mask = np.zeros(len(scene_pts_np), dtype=bool)
        max_k = int(self.params["max_candidates"])
        radius = ctx.model_radius_mm if ctx.model_radius_mm else float(np.linalg.norm(
            np.asarray(model_ds.get_max_bound()) - np.asarray(model_ds.get_min_bound())
        )) * 0.5

        for k in range(max_k):
            if explained_mask.all():
                break
            unexpl = np.where(~explained_mask)[0]
            if len(unexpl) < 30:
                break
            sub = scene_ds.select_by_index(unexpl.tolist())
            # rebuild FPFH for subset: cheaper to select the column indices
            sub_fpfh = o3d.pipelines.registration.Feature()
            sub_fpfh.data = fpfh_s.data[:, unexpl]

            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                model_ds, sub, fpfh_m, sub_fpfh,
                mutual_filter=bool(self.params["mutual_filter"]),
                max_correspondence_distance=vs * 1.5,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=int(self.params["ransac_n"]),
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(vs * 1.5),
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                    int(self.params["ransac_iters"]), float(self.params["ransac_confidence"])
                ),
            )
            if result.fitness < 0.05:
                break
            T = np.asarray(result.transformation)
            c = T[:3, 3]
            d = Detection(
                instance_id=k,
                pose=T,
                confidence=float(result.fitness),
                fitness=float(result.fitness),
                inlier_rmse=float(result.inlier_rmse),
                method="feature_ransac",
                extra={"iteration": k},
            )
            candidates.append(d)
            # mark points within ~radius of placed model as explained
            dists = np.linalg.norm(scene_pts_np - c, axis=1)
            explained_mask |= dists < (radius * 1.5)
            if ctx.progress:
                ctx.progress.emit(0.4 + 0.6 * (k + 1) / max_k, f"global match {k+1}")

        ctx.candidates = candidates
        return {"n_candidates": len(candidates)}

    def _twod_rois(self, ctx: PipelineContext) -> dict:
        """Each 2D detection becomes one 3D ROI → one coarse candidate.

        Lifts the bbox to 3D via the ordered grid, then either RANSAC-fits a
        sphere (for sphere-like models) or places an identity-rotation seed at
        the ROI centroid.
        """
        if ctx.scene_ordered_grid is None:
            raise RuntimeError("twod_rois requires an ordered-grid scene.")
        if not ctx.twod_detections:
            raise RuntimeError(
                "twod_rois needs 2D detections — run the YOLO-World detector "
                "in the '2D RGB' tab first."
            )
        g = ctx.scene_ordered_grid
        xyz_hw = g["xyz_hw"]
        valid_hw = g["valid_mask_hw"]
        H, W = g["shape"]
        dil = int(self.params.get("roi_bbox_dilate_px", 0))
        dets: list[Detection] = []
        cluster_label_img = np.full((H, W), -1, dtype=np.int32)

        for i, d2 in enumerate(ctx.twod_detections):
            if getattr(d2, "mask", None) is not None:
                mask_hw = d2.mask
            else:
                x1, y1, x2, y2 = d2.bbox
                x1 = max(0, int(np.floor(x1)) - dil)
                y1 = max(0, int(np.floor(y1)) - dil)
                x2 = min(W, int(np.ceil(x2)) + dil)
                y2 = min(H, int(np.ceil(y2)) + dil)
                mask_hw = np.zeros((H, W), dtype=bool)
                mask_hw[y1:y2, x1:x2] = True
            roi_mask = mask_hw & valid_hw
            cluster_label_img[roi_mask] = i
            roi_pts = xyz_hw[roi_mask].astype(np.float32)
            if len(roi_pts) < int(self.params["roi_min_points"]):
                continue

            pose = np.eye(4)
            extra = {
                "twod_class": d2.class_name,
                "twod_conf": float(d2.confidence),
                "twod_bbox": [float(x) for x in d2.bbox],
                "roi_n_points": int(len(roi_pts)),
            }
            # If the model is sphere-like, use the fast sphere RANSAC fit —
            # it gives a translation with ~mm accuracy and makes rotation moot.
            # Otherwise, use PCA principal-axis alignment against the STL's
            # sample points as the rotation seed for downstream ICP.
            if ctx.model_radius_mm is not None:
                res = _ransac_sphere(
                    roi_pts,
                    radius_prior_mm=ctx.model_radius_mm,
                    inlier_tol_mm=float(self.params["roi_sphere_inlier_tol_mm"]),
                    iterations=int(self.params["roi_sphere_ransac_iters"]),
                    radius_tolerance_frac=float(self.params.get("radius_tolerance_frac", 0.15)),
                    min_inliers=int(self.params["roi_min_points"]) // 2,
                    seed=2000 + i,
                )
                if res is not None:
                    c, r, inl = res
                    pose[:3, 3] = c
                    extra.update({
                        "fitted_radius_mm": float(r),
                        "radius_error_mm": float(abs(r - ctx.model_radius_mm)),
                        "sphere_inliers": int(inl.sum()),
                        "seed_kind": "sphere_ransac",
                    })
                    conf = float(inl.sum()) / max(len(roi_pts), 1)
                else:
                    pose[:3, 3] = roi_pts.mean(axis=0)
                    extra["seed_kind"] = "centroid"
                    conf = float(d2.confidence)
            else:
                T_seed, cost = _pca_align(roi_pts, ctx.model_points)
                pose = T_seed
                extra.update({
                    "seed_kind": "pca_align",
                    "pca_chamfer_mean_mm": float(cost),
                })
                conf = float(d2.confidence)

            dets.append(Detection(
                instance_id=len(dets),
                pose=pose,
                confidence=conf,
                fitness=conf,
                method="twod_roi",
                n_inliers=int(extra.get("sphere_inliers", 0)),
                extra=extra,
            ))

        ctx.candidates = dets
        # Project cluster labels from image to current_points for viz.
        if len(ctx.current_points):
            tree = cKDTree(ctx.scene_points)
            _, nn_idx = tree.query(ctx.current_points, k=1)
            pix = g["pixel_of_point"]
            u = pix[nn_idx, 0]; v = pix[nn_idx, 1]
            ctx.cluster_labels = cluster_label_img[v, u]
        return {"n_candidates": len(dets), "n_twod_dets": len(ctx.twod_detections)}

    def _twod_feature_rois(self, ctx: PipelineContext) -> dict:
        """Per-2D-ROI feature (FPFH) matching.

        For each 2D detection:
          1. Lift bbox → 3D ROI points via ordered grid.
          2. Voxel-downsample ROI, compute FPFH, run feature-matching RANSAC
             against the STL sample's FPFH.
          3. Emit one Detection per ROI with the RANSAC transform as seed.

        Strong for asymmetric parts (bolts, cogs, etc). Fails on symmetric
        parts by design, same as the global `feature_ransac` method.
        """
        if ctx.scene_ordered_grid is None:
            raise RuntimeError("twod_feature_rois requires an ordered-grid scene.")
        if not ctx.twod_detections:
            raise RuntimeError("twod_feature_rois needs 2D detections — run the detector first.")
        g = ctx.scene_ordered_grid
        xyz_hw = g["xyz_hw"]
        valid_hw = g["valid_mask_hw"]
        H, W = g["shape"]
        vs = float(self.params["voxel_scene_mm"])
        vm = float(self.params["voxel_model_mm"])
        fr = float(self.params["feature_radius_mul"]) * max(vs, vm)

        model_pcd = o3d.geometry.PointCloud()
        model_pcd.points = o3d.utility.Vector3dVector(ctx.model_points)
        model_ds = model_pcd.voxel_down_sample(vm)
        model_ds.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=vm * 2.0, max_nn=30))
        fpfh_m = o3d.pipelines.registration.compute_fpfh_feature(
            model_ds, o3d.geometry.KDTreeSearchParamHybrid(radius=fr, max_nn=100))

        dil = int(self.params.get("roi_bbox_dilate_px", 0))
        dets: list[Detection] = []
        for i, d2 in enumerate(ctx.twod_detections):
            x1, y1, x2, y2 = d2.bbox
            x1 = max(0, int(np.floor(x1)) - dil)
            y1 = max(0, int(np.floor(y1)) - dil)
            x2 = min(W, int(np.ceil(x2)) + dil)
            y2 = min(H, int(np.ceil(y2)) + dil)
            mask_hw = np.zeros((H, W), dtype=bool)
            mask_hw[y1:y2, x1:x2] = True
            roi_mask = mask_hw & valid_hw
            roi_pts = xyz_hw[roi_mask].astype(np.float32)
            if len(roi_pts) < int(self.params["roi_min_points"]):
                continue
            scene_pcd = o3d.geometry.PointCloud()
            scene_pcd.points = o3d.utility.Vector3dVector(roi_pts)
            scene_ds = scene_pcd.voxel_down_sample(vs)
            scene_ds.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=vs * 2.0, max_nn=30))
            if len(scene_ds.points) < int(self.params["roi_min_points"]) // 3:
                continue
            fpfh_s = o3d.pipelines.registration.compute_fpfh_feature(
                scene_ds, o3d.geometry.KDTreeSearchParamHybrid(radius=fr, max_nn=100))
            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                model_ds, scene_ds, fpfh_m, fpfh_s,
                mutual_filter=bool(self.params["mutual_filter"]),
                max_correspondence_distance=vs * 1.5,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=int(self.params["ransac_n"]),
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(vs * 1.5),
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                    int(self.params["ransac_iters"]),
                    float(self.params["ransac_confidence"])),
            )
            T = np.asarray(result.transformation)
            dets.append(Detection(
                instance_id=i,
                pose=T,
                confidence=float(result.fitness),
                fitness=float(result.fitness),
                inlier_rmse=float(result.inlier_rmse),
                method="twod_feature_roi",
                extra={
                    "twod_class": d2.class_name,
                    "twod_conf": float(d2.confidence),
                    "twod_bbox": [float(x) for x in d2.bbox],
                    "roi_n_points": int(len(roi_pts)),
                    "seed_kind": "fpfh_ransac",
                },
            ))
            if ctx.progress:
                ctx.progress.emit((i + 1) / max(len(ctx.twod_detections), 1),
                                    f"FPFH in ROI {i+1}/{len(ctx.twod_detections)}")
        ctx.candidates = dets
        return {"n_candidates": len(dets), "n_twod_dets": len(ctx.twod_detections)}

    def run(self, ctx: PipelineContext) -> StageResult:
        pts = ctx.current_points
        n_in = len(pts)
        stats: dict
        if self.method == "dbscan":
            stats = self._dbscan(ctx)
        elif self.method == "sphere_ransac_per_cluster":
            stats = self._sphere_per_cluster(ctx)
        elif self.method == "feature_ransac":
            stats = self._feature_ransac(ctx)
        elif self.method == "cluster_centers":
            stats = self._cluster_centers(ctx)
        elif self.method == "twod_rois":
            stats = self._twod_rois(ctx)
        elif self.method == "twod_feature_rois":
            stats = self._twod_feature_rois(ctx)
        else:
            raise ValueError(f"unknown candidates method {self.method}")

        return StageResult(
            slot=self.slot,
            method=self.method,
            params=self.params,
            n_in=n_in,
            n_out=len(ctx.candidates),
            duration_s=0.0,
            stats=stats,
        )
