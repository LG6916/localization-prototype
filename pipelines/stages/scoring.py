"""Scoring and non-maximum suppression of detections."""
from __future__ import annotations

import numpy as np

from ..base import Stage, StageResult, PipelineContext


class ScoringStage(Stage):
    slot = "scoring"

    METHODS = ["standard", "pose_nms", "passthrough"]

    def __init__(self, method: str = "standard", params=None):
        defaults = dict(
            min_fitness=0.15,
            nms_distance_mm=20.0,
            nms_use_model_diag=True,
            max_results=50,
        )
        defaults.update(params or {})
        super().__init__(defaults)
        self.method = method

    def run(self, ctx: PipelineContext) -> StageResult:
        n_in = len(ctx.detections)
        if self.method == "passthrough":
            kept = list(ctx.detections)
        else:
            dets = list(ctx.detections)
            dets.sort(key=lambda d: d.confidence, reverse=True)

            # threshold
            if self.method in ("standard", "pose_nms"):
                min_f = float(self.params["min_fitness"])
                dets = [d for d in dets if d.confidence >= min_f]

            # NMS by translation distance
            nms_d = float(self.params["nms_distance_mm"])
            if self.params.get("nms_use_model_diag", True) and ctx.model_points is not None:
                diag = float(np.linalg.norm(
                    ctx.model_points.max(0) - ctx.model_points.min(0)
                ))
                nms_d = max(nms_d, 0.5 * diag)

            kept = []
            for d in dets:
                c = d.translation
                dup = any(np.linalg.norm(c - k.translation) < nms_d for k in kept)
                if not dup:
                    kept.append(d)
                if len(kept) >= int(self.params["max_results"]):
                    break
            # re-id
            for i, d in enumerate(kept):
                d.instance_id = i

        ctx.detections = kept
        return StageResult(
            slot=self.slot, method=self.method, params=self.params,
            n_in=n_in, n_out=len(kept), duration_s=0.0,
            stats={
                "kept_fraction": len(kept) / max(n_in, 1),
                "min_fitness": float(self.params["min_fitness"]),
            },
        )
