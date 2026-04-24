"""Per-point colorization helpers."""
from __future__ import annotations

import numpy as np


def fit_quality_colors(distances_mm: np.ndarray, good_mm: float = 1.0, bad_mm: float = 6.0) -> np.ndarray:
    """Map point-to-model distances to RGB 0..1. Green=good, Yellow=mid, Red=bad."""
    t = np.clip((distances_mm - good_mm) / max(bad_mm - good_mm, 1e-6), 0.0, 1.0)
    r = t
    g = 1.0 - t
    b = np.zeros_like(t)
    return np.stack([r, g, b], axis=1)


def segment_colors(labels: np.ndarray) -> np.ndarray:
    """Stable pseudo-random color per integer label. -1 -> gray."""
    out = np.zeros((len(labels), 3), dtype=np.float32)
    rng = np.random.default_rng(42)
    palette = rng.random((256, 3)).astype(np.float32)
    # Mute saturation a bit
    palette = 0.2 + 0.7 * palette
    for i, l in enumerate(labels):
        if l < 0:
            out[i] = (0.3, 0.3, 0.3)
        else:
            out[i] = palette[int(l) % 256]
    return out
