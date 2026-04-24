"""Preprocess stage: voxel downsample, outlier removal, normal estimation."""
from __future__ import annotations

import numpy as np
import open3d as o3d

from ..base import Stage, StageResult, PipelineContext


def _to_open3d(points, colors=None, normals=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


class PreprocessStage(Stage):
    slot = "preprocess"

    METHODS = ["voxel+outlier", "voxel_only", "outlier_only", "passthrough"]

    def __init__(self, method: str = "voxel+outlier", params=None):
        defaults = dict(
            voxel_mm=3.0,
            stat_nb_neighbors=20,
            stat_std_ratio=2.0,
            estimate_normals=True,
            normal_radius_mm=6.0,
            normal_max_nn=30,
        )
        defaults.update(params or {})
        super().__init__(defaults)
        self.method = method

    def run(self, ctx: PipelineContext) -> StageResult:
        pts = ctx.scene_points
        cols = ctx.scene_colors
        n_in = len(pts)
        pcd = _to_open3d(pts, cols)

        if self.method in ("voxel+outlier", "voxel_only"):
            pcd = pcd.voxel_down_sample(float(self.params["voxel_mm"]))
            if ctx.progress: ctx.progress.emit(0.4, "voxel downsample")

        if self.method in ("voxel+outlier", "outlier_only"):
            pcd, _ = pcd.remove_statistical_outlier(
                int(self.params["stat_nb_neighbors"]),
                float(self.params["stat_std_ratio"]),
            )
            if ctx.progress: ctx.progress.emit(0.7, "statistical outlier removal")

        if self.params.get("estimate_normals", True):
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=float(self.params["normal_radius_mm"]),
                    max_nn=int(self.params["normal_max_nn"]),
                )
            )
            pcd.orient_normals_consistent_tangent_plane(k=15) if False else None

        out_pts = np.asarray(pcd.points, dtype=np.float32)
        out_cols = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
        out_nrm = np.asarray(pcd.normals, dtype=np.float32) if pcd.has_normals() else None

        ctx.current_points = out_pts
        ctx.current_colors = out_cols
        ctx.current_normals = out_nrm

        return StageResult(
            slot=self.slot,
            method=self.method,
            params=self.params,
            n_in=n_in,
            n_out=len(out_pts),
            duration_s=0.0,
            stats={
                "kept_fraction": len(out_pts) / max(n_in, 1),
                "has_normals": out_nrm is not None,
            },
            viz={"points_kept": out_pts.shape[0]},
        )
