"""STL/OBJ model loading with unit auto-detection and surface sampling."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh


@dataclass
class ModelData:
    mesh: trimesh.Trimesh
    unit_scale_to_mm: float = 1.0
    auto_detected_scale: Optional[float] = None
    sphere_radius_mm: Optional[float] = None   # if the mesh is sphere-like
    sphericity: float = 0.0                    # 1.0 = perfect sphere
    source_path: Optional[str] = None
    meta: dict = field(default_factory=dict)

    @property
    def extents_mm(self) -> np.ndarray:
        return self.mesh.extents * self.unit_scale_to_mm

    @property
    def centroid_mm(self) -> np.ndarray:
        return self.mesh.centroid * self.unit_scale_to_mm

    def sample_points(self, n: int = 3000, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        pts, _ = trimesh.sample.sample_surface(self.mesh, n, seed=int(rng.integers(0, 2**31 - 1)))
        return np.asarray(pts, dtype=np.float32) * self.unit_scale_to_mm


def _infer_sphere_params(mesh: trimesh.Trimesh) -> tuple[Optional[float], float]:
    """If mesh looks like a sphere, return (radius_in_mesh_units, sphericity 0..1)."""
    if len(mesh.vertices) < 50:
        return None, 0.0
    c = mesh.centroid
    d = np.linalg.norm(mesh.vertices - c, axis=1)
    if d.mean() < 1e-9:
        return None, 0.0
    cv = d.std() / d.mean()
    sphericity = float(max(0.0, 1.0 - cv * 10))  # squishes cv into [0,1]
    if cv < 0.02:
        return float(d.mean()), sphericity
    return None, sphericity


def load_model(
    path: str | Path,
    *,
    scene_bbox_mm: Optional[np.ndarray] = None,
    unit_override: Optional[float] = None,
) -> ModelData:
    """Load an STL/OBJ model.

    If `unit_override` is None, tries to auto-detect whether the mesh is in meters
    (extent < 10) vs millimeters. Heuristic: if max extent < 10 mm AND scene extent
    > 100 mm, treat model as meters (x1000).
    """
    path = Path(path)
    mesh = trimesh.load(str(path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])

    auto_scale = None
    if unit_override is not None:
        scale = float(unit_override)
    else:
        max_ext = float(mesh.extents.max())
        scene_big = scene_bbox_mm is not None and float(np.max(scene_bbox_mm)) > 100
        if max_ext < 1.0 and scene_big:
            scale = 1000.0          # meters -> mm
            auto_scale = scale
        elif max_ext < 10.0 and scene_big:
            # ambiguous small model (cm?) — most CAD is mm, but sphere STL was meters
            scale = 1000.0 if max_ext < 1.0 else 1.0
            auto_scale = scale
        else:
            scale = 1.0

    radius_src, sphericity = _infer_sphere_params(mesh)
    radius_mm = radius_src * scale if radius_src is not None else None

    return ModelData(
        mesh=mesh,
        unit_scale_to_mm=scale,
        auto_detected_scale=auto_scale,
        sphere_radius_mm=radius_mm,
        sphericity=sphericity,
        source_path=str(path),
        meta={"triangles": int(len(mesh.faces)), "vertices": int(len(mesh.vertices))},
    )
