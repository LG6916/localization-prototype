"""Core pipeline abstractions.

A Pipeline is an ordered list of Stages. Each Stage reads and writes a
PipelineContext. Stages report timings / intermediate stats through a
StageResult so the UI can render a detailed pipeline trace.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# ----- Detection type ---------------------------------------------------------


@dataclass
class Detection:
    """A single localized object instance."""
    instance_id: int
    pose: np.ndarray               # 4x4 transform from model frame -> scene frame (mm)
    confidence: float              # 0..1, algorithm-specific
    fitness: float = 0.0           # Open3D ICP fitness (inlier fraction)
    inlier_rmse: float = 0.0       # Open3D ICP inlier RMSE (mm)
    n_inliers: int = 0
    method: str = ""               # "sphere_ransac" | "icp_p2pl" | ...
    extra: dict = field(default_factory=dict)   # e.g. radius_err, cluster_id

    @property
    def translation(self) -> np.ndarray:
        return self.pose[:3, 3].copy()

    @property
    def rotation(self) -> np.ndarray:
        return self.pose[:3, :3].copy()


# ----- Progress ---------------------------------------------------------------


class ProgressReporter:
    """Minimal progress sink. Wraps a callable that accepts (label, fraction)."""

    def __init__(self, sink: Optional[Callable[[str, float], None]] = None):
        self._sink = sink
        self._stage_base = 0.0
        self._stage_span = 1.0
        self._stage_label = ""

    def begin_stage(self, label: str, base: float, span: float):
        self._stage_label = label
        self._stage_base = base
        self._stage_span = span
        self.emit(0.0, label)

    def emit(self, local_frac: float, sub_label: Optional[str] = None):
        if self._sink is None:
            return
        frac = self._stage_base + max(0.0, min(1.0, local_frac)) * self._stage_span
        text = sub_label or self._stage_label
        try:
            self._sink(text, frac)
        except Exception:
            pass


# ----- Context ----------------------------------------------------------------


@dataclass
class PipelineContext:
    """Shared mutable state passed between stages."""
    scene_points: np.ndarray
    scene_colors: Optional[np.ndarray]
    model_points: np.ndarray           # sampled from STL surface, in mm
    model_mesh: object = None          # trimesh mesh (mm)
    model_radius_mm: Optional[float] = None   # present for sphere-like models

    # Mutable working state
    current_points: np.ndarray = field(default=None)
    current_colors: Optional[np.ndarray] = None
    current_normals: Optional[np.ndarray] = None
    removed_points: Optional[np.ndarray] = None
    removed_labels: Optional[np.ndarray] = None   # which background strategy flagged each removed pt
    plane_model: Optional[tuple] = None           # (a,b,c,d) if a plane was fit

    cluster_labels: Optional[np.ndarray] = None   # per-point cluster id for current_points
    candidates: list = field(default_factory=list)   # list[Detection] (coarse)
    detections: list = field(default_factory=list)   # list[Detection] (final)

    # Per-stage viz snapshots (list of dicts); UI reads these to render layers
    trace: list = field(default_factory=list)

    # Optional 2D-guided inputs (populated when the user ran the 2D detector).
    # Structure of scene_ordered_grid:
    #   {"xyz_hw": (H,W,3) float32, "valid_mask_hw": (H,W) bool,
    #    "pixel_of_point": (N,2) int32, "point_of_pixel": (H,W) int32 (-1=invalid),
    #    "shape": (H, W)}
    scene_ordered_grid: Optional[dict] = None
    twod_detections: list = field(default_factory=list)   # list[Detection2D]

    # Global config
    progress: Optional[ProgressReporter] = None


# ----- Stage / Result ---------------------------------------------------------


@dataclass
class StageResult:
    slot: str
    method: str
    params: dict
    n_in: int
    n_out: int
    duration_s: float
    stats: dict = field(default_factory=dict)
    viz: dict = field(default_factory=dict)     # optional layer payload for this stage


class Stage:
    slot: str = ""     # 'preprocess'|'background'|'candidates'|'refine'|'scoring'
    method: str = ""

    def __init__(self, params: Optional[dict] = None):
        self.params = dict(params or {})

    def run(self, ctx: PipelineContext) -> StageResult:
        raise NotImplementedError


# ----- Pipeline ---------------------------------------------------------------


class Pipeline:
    def __init__(self, name: str, stages: list[Stage]):
        self.name = name
        self.stages = stages

    def run(self, ctx: PipelineContext) -> list[StageResult]:
        results: list[StageResult] = []
        n = len(self.stages)
        for i, stg in enumerate(self.stages):
            base = i / max(n, 1)
            span = 1.0 / max(n, 1)
            if ctx.progress is not None:
                ctx.progress.begin_stage(f"{stg.slot}: {stg.method}", base, span)
            t0 = time.time()
            res = stg.run(ctx)
            res.duration_s = time.time() - t0
            results.append(res)
            ctx.trace.append(res)
            if ctx.progress is not None:
                ctx.progress.emit(1.0)
        return results
