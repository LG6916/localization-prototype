"""Voxel-based display downsampling so the browser stays responsive."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d


@dataclass
class DisplayCopy:
    points: np.ndarray
    colors: Optional[np.ndarray]
    voxel_mm: float
    source_n: int


def build_display_copy(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    *,
    max_points: int = 50_000,
    min_voxel_mm: float = 0.2,
) -> DisplayCopy:
    """Produce a voxel-downsampled rendering copy under `max_points`.

    Binary-searches voxel size until point count <= max_points. Preserves colors.
    """
    n = len(points)
    if n <= max_points:
        return DisplayCopy(points=points.astype(np.float32),
                           colors=colors if colors is None else colors.astype(np.float32),
                           voxel_mm=0.0, source_n=n)

    extents = points.max(0) - points.min(0)
    diag = float(np.linalg.norm(extents))

    lo, hi = min_voxel_mm, max(min_voxel_mm * 4, diag / 50)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    best: Optional[o3d.geometry.PointCloud] = None
    best_v = hi
    # expand upper bound if needed
    for _ in range(10):
        ds = pcd.voxel_down_sample(hi)
        if len(ds.points) <= max_points:
            break
        hi *= 1.7
    # bisect
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        ds = pcd.voxel_down_sample(mid)
        if len(ds.points) <= max_points:
            best, best_v = ds, mid
            hi = mid
        else:
            lo = mid
    if best is None:
        best = pcd.voxel_down_sample(hi)
        best_v = hi

    out_pts = np.asarray(best.points, dtype=np.float32)
    out_cols = np.asarray(best.colors, dtype=np.float32) if best.has_colors() else None
    return DisplayCopy(points=out_pts, colors=out_cols, voxel_mm=float(best_v), source_n=n)
