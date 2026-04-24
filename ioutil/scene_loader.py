"""PLY scene loading with Photoneo metadata awareness."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d


@dataclass
class SceneData:
    points: np.ndarray                  # (N, 3) mm, filtered valid
    colors: Optional[np.ndarray] = None # (N, 3) 0..1, filtered valid
    normals: Optional[np.ndarray] = None
    source_path: Optional[str] = None
    photoneo_grid: Optional[tuple[int, int]] = None   # (W, H) if ordered
    meta: dict = field(default_factory=dict)

    # Ordered-grid representation — only populated when PLY is ordered.
    # Arrays are indexed [row (v), col (u), channel]; u is the horizontal axis.
    xyz_hw: Optional[np.ndarray] = None            # (H, W, 3) float32; (0,0,0) at invalid pixels
    rgb_hw: Optional[np.ndarray] = None            # (H, W, 3) uint8
    valid_mask_hw: Optional[np.ndarray] = None     # (H, W) bool
    pixel_of_point: Optional[np.ndarray] = None    # (N, 2) int32: (u, v) per filtered point
    point_of_pixel: Optional[np.ndarray] = None    # (H, W) int32; -1 where invalid

    def to_open3d(self) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        if self.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(self.colors)
        if self.normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(self.normals)
        return pcd

    @property
    def is_ordered(self) -> bool:
        return self.xyz_hw is not None


def _parse_photoneo_header(path: Path) -> dict:
    """Extract Photoneo obj_info Width/Height if present. Non-fatal."""
    meta: dict = {}
    try:
        with open(path, "rb") as f:
            chunk = f.read(4096)
        text = chunk.decode(errors="replace")
        for line in text.splitlines():
            if line.startswith("obj_info"):
                if "Width" in line and "Height" in line:
                    import re
                    m = re.search(r"Width\s*=\s*(\d+).*Height\s*=\s*(\d+)", line)
                    if m:
                        meta["grid"] = (int(m.group(1)), int(m.group(2)))
                        meta["ordered"] = "Ordered" in line
    except Exception:
        pass
    return meta


def load_scene(path: str | Path, *, drop_invalid: bool = True) -> SceneData:
    """Load a PLY scene.

    Drops (0,0,0) invalid returns by default (common on structured-light scans).
    If the PLY header advertises an ordered grid (e.g. Photoneo), we also
    populate `xyz_hw` / `rgb_hw` / `valid_mask_hw` and index maps so downstream
    code can lift 2D regions back to 3D for free.

    Returns SceneData in whatever units the file is in — typically millimeters
    for Photoneo scans.
    """
    path = Path(path)
    meta = _parse_photoneo_header(path)

    pcd = o3d.io.read_point_cloud(str(path))
    pts_all = np.asarray(pcd.points, dtype=np.float32)
    cols_all = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
    nrm_all = np.asarray(pcd.normals, dtype=np.float32) if pcd.has_normals() else None

    # Ordered-grid reshape BEFORE any filtering, while array order still matches
    # pixel-major (v * W + u) layout as written in the PLY.
    grid = meta.get("grid")   # (W, H)
    xyz_hw = rgb_hw = valid_mask_hw = pixel_of_point = point_of_pixel = None
    if grid is not None and len(pts_all) == grid[0] * grid[1]:
        W, H = grid
        xyz_hw = pts_all.reshape(H, W, 3).copy()
        valid_mask_hw = np.linalg.norm(xyz_hw, axis=2) > 1e-6
        if cols_all is not None:
            rgb_hw = (np.clip(cols_all, 0.0, 1.0) * 255).astype(np.uint8).reshape(H, W, 3)
        else:
            # Fallback: fake RGB from depth shading so the 2D tab still has something
            z = xyz_hw[..., 2]
            zn = np.zeros_like(z) if not valid_mask_hw.any() else (
                np.clip((z - z[valid_mask_hw].min()) /
                        max(z[valid_mask_hw].ptp(), 1e-6), 0, 1)
            )
            gray = (zn * 255).astype(np.uint8)
            rgb_hw = np.stack([gray, gray, gray], axis=2)

    if drop_invalid and len(pts_all):
        valid = np.linalg.norm(pts_all, axis=1) > 1e-6
    else:
        valid = np.ones(len(pts_all), dtype=bool)

    pts = pts_all[valid]
    cols = cols_all[valid] if cols_all is not None else None
    nrm = nrm_all[valid] if nrm_all is not None else None

    # Build pixel <-> filtered-index maps when we have a grid.
    if xyz_hw is not None:
        W, H = grid
        flat_idx = np.arange(W * H, dtype=np.int32)
        kept_idx = flat_idx[valid]
        u = (kept_idx % W).astype(np.int32)
        v = (kept_idx // W).astype(np.int32)
        pixel_of_point = np.stack([u, v], axis=1)
        point_of_pixel = np.full((H, W), -1, dtype=np.int32)
        point_of_pixel[v, u] = np.arange(len(kept_idx), dtype=np.int32)

    return SceneData(
        points=pts,
        colors=cols,
        normals=nrm,
        source_path=str(path),
        photoneo_grid=grid,
        meta=meta,
        xyz_hw=xyz_hw,
        rgb_hw=rgb_hw,
        valid_mask_hw=valid_mask_hw,
        pixel_of_point=pixel_of_point,
        point_of_pixel=point_of_pixel,
    )
