"""Pose refinement: ICP variants, or none (pass candidates through)."""
from __future__ import annotations

import numpy as np
import open3d as o3d

from ..base import Stage, StageResult, PipelineContext, Detection


def _to_o3d(points, colors=None, normals=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


class RefineStage(Stage):
    slot = "refine"

    METHODS = ["icp_p2p", "icp_p2pl", "icp_multiscale", "icp_colored", "none"]

    def __init__(self, method: str = "icp_p2pl", params=None):
        defaults = dict(
            max_corr_dist_mm=4.0,
            max_iter=60,
            multi_scale_voxels_mm=(8.0, 4.0, 2.0),
            multi_scale_iters=(40, 30, 20),
            # for cluster-seeded ICP multi-start
            multi_start=True,
            n_rotations=8,
            # colored ICP
            colored_lambda_geom=0.968,
        )
        defaults.update(params or {})
        super().__init__(defaults)
        self.method = method

    def _icp_once(self, scene_pcd, model_pcd, init_T, max_dist, max_iter, use_normals):
        est = (
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
            if use_normals
            else o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
        return o3d.pipelines.registration.registration_icp(
            model_pcd, scene_pcd, max_dist, init_T, est, crit
        )

    def run(self, ctx: PipelineContext) -> StageResult:
        if self.method == "none" or not ctx.candidates:
            ctx.detections = list(ctx.candidates)
            return StageResult(
                slot=self.slot, method="none", params=self.params,
                n_in=len(ctx.candidates), n_out=len(ctx.detections),
                duration_s=0.0, stats={"skipped": True},
            )

        scene_pcd = _to_o3d(ctx.current_points, ctx.current_colors, ctx.current_normals)
        if not scene_pcd.has_normals():
            scene_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=6.0, max_nn=30))

        model_pts = ctx.model_points
        model_pcd = _to_o3d(model_pts)
        model_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=6.0, max_nn=30))

        out: list[Detection] = []
        n = len(ctx.candidates)
        for i, cand in enumerate(ctx.candidates):
            init_T = cand.pose.copy()
            tries = [init_T]
            if self.params.get("multi_start", True) and np.allclose(init_T[:3, :3], np.eye(3)):
                # rotate around z axis as multi-start
                k = int(self.params["n_rotations"])
                for j in range(1, k):
                    a = 2 * np.pi * j / k
                    R = np.array([
                        [np.cos(a), -np.sin(a), 0],
                        [np.sin(a),  np.cos(a), 0],
                        [0, 0, 1],
                    ])
                    T = init_T.copy()
                    T[:3, :3] = R
                    tries.append(T)
            best = None
            for T0 in tries:
                if self.method == "icp_p2p":
                    r = self._icp_once(scene_pcd, model_pcd, T0,
                                       float(self.params["max_corr_dist_mm"]),
                                       int(self.params["max_iter"]),
                                       use_normals=False)
                elif self.method == "icp_p2pl":
                    r = self._icp_once(scene_pcd, model_pcd, T0,
                                       float(self.params["max_corr_dist_mm"]),
                                       int(self.params["max_iter"]),
                                       use_normals=True)
                elif self.method == "icp_colored":
                    # Colored ICP needs colors on both clouds. The model is sampled
                    # from STL (no color), so we paint it a neutral gray and rely on
                    # the scene's color gradient for alignment. Downgrade to p2pl if
                    # scene has no colors.
                    if not scene_pcd.has_colors():
                        r = self._icp_once(scene_pcd, model_pcd, T0,
                                            float(self.params["max_corr_dist_mm"]),
                                            int(self.params["max_iter"]),
                                            use_normals=True)
                    else:
                        if not model_pcd.has_colors():
                            model_pcd.paint_uniform_color([0.5, 0.5, 0.5])
                        voxels = self.params.get("multi_scale_voxels_mm", (8.0, 4.0, 2.0))
                        iters = self.params.get("multi_scale_iters", (40, 30, 20))
                        cur_T = T0
                        for v, it in zip(voxels, iters):
                            s_ds = scene_pcd.voxel_down_sample(float(v))
                            if not s_ds.has_normals():
                                s_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v*2, max_nn=30))
                            m_ds = model_pcd.voxel_down_sample(float(v))
                            if not m_ds.has_normals():
                                m_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v*2, max_nn=30))
                            r = o3d.pipelines.registration.registration_colored_icp(
                                m_ds, s_ds, float(v) * 2.0, cur_T,
                                o3d.pipelines.registration.TransformationEstimationForColoredICP(
                                    lambda_geometric=float(self.params.get("colored_lambda_geom", 0.968))
                                ),
                                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(it)),
                            )
                            cur_T = r.transformation
                elif self.method == "icp_multiscale":
                    voxels = self.params["multi_scale_voxels_mm"]
                    iters = self.params["multi_scale_iters"]
                    cur_T = T0
                    for v, it in zip(voxels, iters):
                        s_ds = scene_pcd.voxel_down_sample(float(v))
                        if not s_ds.has_normals():
                            s_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v*2, max_nn=30))
                        m_ds = model_pcd.voxel_down_sample(float(v))
                        if not m_ds.has_normals():
                            m_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v*2, max_nn=30))
                        r = o3d.pipelines.registration.registration_icp(
                            m_ds, s_ds, float(v) * 2.0, cur_T,
                            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(it)),
                        )
                        cur_T = r.transformation
                else:
                    raise ValueError(f"unknown refine method {self.method}")
                if best is None or r.fitness > best.fitness:
                    best = r

            T = np.asarray(best.transformation)
            det = Detection(
                instance_id=i,
                pose=T,
                confidence=float(best.fitness),
                fitness=float(best.fitness),
                inlier_rmse=float(best.inlier_rmse),
                n_inliers=int(round(best.fitness * len(model_pcd.points))),
                method=self.method,
                extra={**cand.extra, "seed_method": cand.method},
            )
            out.append(det)
            if ctx.progress:
                ctx.progress.emit((i + 1) / max(n, 1), f"ICP {i+1}/{n}")

        ctx.detections = out
        return StageResult(
            slot=self.slot, method=self.method, params=self.params,
            n_in=n, n_out=len(out), duration_s=0.0,
            stats={
                "mean_fitness": float(np.mean([d.fitness for d in out])) if out else 0.0,
                "mean_rmse_mm": float(np.mean([d.inlier_rmse for d in out])) if out else 0.0,
            },
        )
