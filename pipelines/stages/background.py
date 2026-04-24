"""Background removal: plane RANSAC or depth cutoff."""
from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from ..base import Stage, StageResult, PipelineContext


class BackgroundStage(Stage):
    slot = "background"

    METHODS = ["plane_ransac", "plane_ransac_multi", "depth_cutoff",
               "twod_mask_foreground", "none"]

    def __init__(self, method: str = "plane_ransac", params=None):
        defaults = dict(
            distance_threshold_mm=3.0,
            ransac_n=3,
            num_iterations=1000,
            min_plane_fraction=0.2,
            max_planes=2,
            depth_z_min_mm=-1e9,
            depth_z_max_mm=1e9,
            # 2D-guided foreground
            bbox_dilate_px=8,
            require_ordered_grid=True,
        )
        defaults.update(params or {})
        super().__init__(defaults)
        self.method = method

    def _plane_ransac(self, ctx: PipelineContext, multi: bool) -> tuple[np.ndarray, np.ndarray, list]:
        pts = ctx.current_points
        cols = ctx.current_colors
        nrm = ctx.current_normals
        removed_idx = np.zeros(len(pts), dtype=bool)
        planes = []
        n_target = 1 if not multi else int(self.params["max_planes"])

        working = np.arange(len(pts))
        for i in range(n_target):
            if len(working) < 100:
                break
            sub = o3d.geometry.PointCloud()
            sub.points = o3d.utility.Vector3dVector(pts[working])
            model, inl = sub.segment_plane(
                distance_threshold=float(self.params["distance_threshold_mm"]),
                ransac_n=int(self.params["ransac_n"]),
                num_iterations=int(self.params["num_iterations"]),
            )
            frac = len(inl) / max(len(working), 1)
            if frac < float(self.params["min_plane_fraction"]):
                break
            global_inl = working[inl]
            removed_idx[global_inl] = True
            planes.append(tuple(float(x) for x in model))
            working = np.setdiff1d(working, global_inl, assume_unique=False)
            if ctx.progress: ctx.progress.emit((i + 1) / max(n_target, 1), f"plane {i+1}")

        keep = ~removed_idx
        ctx.current_points = pts[keep]
        if cols is not None:
            ctx.current_colors = cols[keep]
        if nrm is not None:
            ctx.current_normals = nrm[keep]
        ctx.removed_points = pts[removed_idx]
        ctx.removed_labels = np.full(int(removed_idx.sum()), "plane", dtype=object)
        ctx.plane_model = planes[0] if planes else None
        return ctx.current_points, ctx.removed_points, planes

    def _depth_cutoff(self, ctx: PipelineContext) -> tuple[np.ndarray, np.ndarray]:
        pts = ctx.current_points
        cols = ctx.current_colors
        nrm = ctx.current_normals
        z = pts[:, 2]
        keep = (z >= float(self.params["depth_z_min_mm"])) & (z <= float(self.params["depth_z_max_mm"]))
        removed = pts[~keep]
        ctx.current_points = pts[keep]
        if cols is not None:
            ctx.current_colors = cols[keep]
        if nrm is not None:
            ctx.current_normals = nrm[keep]
        ctx.removed_points = removed
        ctx.removed_labels = np.full(len(removed), "depth_cutoff", dtype=object)
        return ctx.current_points, removed

    def _twod_mask_foreground(self, ctx: PipelineContext) -> tuple[np.ndarray, np.ndarray, int]:
        """Keep only 3D points whose corresponding pixel lies inside any 2D
        detection bbox (dilated). Requires ordered-grid scene input."""
        if ctx.scene_ordered_grid is None:
            raise RuntimeError(
                "twod_mask_foreground requires an ordered-grid scene "
                "(e.g. Photoneo PLY with obj_info Ordered). Current scene is unordered."
            )
        if not ctx.twod_detections:
            raise RuntimeError(
                "twod_mask_foreground needs 2D detections. Run the YOLO-World "
                "detector in the '2D RGB' tab before clicking Run."
            )
        g = ctx.scene_ordered_grid
        H, W = g["shape"]
        mask = np.zeros((H, W), dtype=bool)
        dil = int(self.params.get("bbox_dilate_px", 0))
        for d in ctx.twod_detections:
            if getattr(d, "mask", None) is not None:
                mask |= d.mask
                continue
            x1, y1, x2, y2 = d.bbox
            x1 = max(0, int(np.floor(x1)) - dil)
            y1 = max(0, int(np.floor(y1)) - dil)
            x2 = min(W, int(np.ceil(x2)) + dil)
            y2 = min(H, int(np.ceil(y2)) + dil)
            mask[y1:y2, x1:x2] = True

        # Project current_points to pixels via nearest neighbor on the full
        # filtered cloud (scene_points). Keep those whose pixel is inside mask.
        orig_pts = ctx.scene_points
        pix = g["pixel_of_point"]      # (N,2) u,v per original filtered point
        tree = cKDTree(orig_pts)
        _, nn_idx = tree.query(ctx.current_points, k=1)
        u = pix[nn_idx, 0]; v = pix[nn_idx, 1]
        keep = mask[v, u]
        removed = ctx.current_points[~keep]
        ctx.current_points = ctx.current_points[keep]
        if ctx.current_colors is not None:
            ctx.current_colors = ctx.current_colors[keep]
        if ctx.current_normals is not None:
            ctx.current_normals = ctx.current_normals[keep]
        ctx.removed_points = removed
        ctx.removed_labels = np.full(len(removed), "2d_background", dtype=object)
        return ctx.current_points, removed, int(mask.sum())

    def run(self, ctx: PipelineContext) -> StageResult:
        pts = ctx.current_points
        n_in = len(pts)
        planes = []
        mask_px = 0
        if self.method == "plane_ransac":
            _, removed, planes = self._plane_ransac(ctx, multi=False)
        elif self.method == "plane_ransac_multi":
            _, removed, planes = self._plane_ransac(ctx, multi=True)
        elif self.method == "depth_cutoff":
            _, removed = self._depth_cutoff(ctx)
        elif self.method == "twod_mask_foreground":
            _, removed, mask_px = self._twod_mask_foreground(ctx)
        else:
            removed = np.zeros((0, 3), dtype=np.float32)
            ctx.removed_points = removed

        n_out = len(ctx.current_points)
        stats = {
            "removed_fraction": (n_in - n_out) / max(n_in, 1),
            "planes_found": len(planes),
            "plane_models": [list(p) for p in planes],
        }
        if mask_px:
            stats["mask_pixels"] = int(mask_px)
            stats["twod_detections"] = len(ctx.twod_detections)
        return StageResult(
            slot=self.slot, method=self.method, params=self.params,
            n_in=n_in, n_out=n_out, duration_s=0.0, stats=stats,
        )
