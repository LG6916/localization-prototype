"""Back-project 3D detection poses to 2D pixel coordinates via the ordered grid.

For Photoneo-style ordered point clouds we already have the pixel-to-3D
mapping baked in — given any 3D point we can find its nearest filtered cloud
sample and read off that sample's (u,v). No camera intrinsics required.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import cKDTree


def build_tree(points: np.ndarray) -> cKDTree:
    return cKDTree(points)


def backproject_points_to_pixels(
    points_3d: np.ndarray,          # (K, 3) query points in scene frame (mm)
    scene_points: np.ndarray,       # (N, 3) filtered scene cloud
    pixel_of_point: np.ndarray,     # (N, 2) int32 (u, v) per filtered point
    *,
    tree: Optional[cKDTree] = None,
    max_dist_mm: float = 25.0,
) -> np.ndarray:
    """Return (K, 2) int array of (u, v) pixels. Rows with no NN closer than
    `max_dist_mm` get (-1, -1).
    """
    if tree is None:
        tree = cKDTree(scene_points)
    dists, idx = tree.query(points_3d, k=1)
    out = pixel_of_point[idx].astype(np.int32).copy()
    out[dists > max_dist_mm] = -1
    return out
