"""RGB preprocessing for dark/low-contrast scanner textures.

CLAHE and min-max normalization, both tuned for Photoneo-style structured-light
RGB (often mean ~50/255). opencv-python is pulled in transitively by
ultralytics, so this only activates when the optional 2D path is installed.
"""
from __future__ import annotations

import numpy as np


def _cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        return None


def apply_clahe(img_rgb: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Apply CLAHE on the L channel of Lab. Returns uint8 RGB."""
    cv2 = _cv2()
    if cv2 is None:
        return img_rgb
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    eq = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    l2 = eq.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)


def apply_normalize(img_rgb: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Min-max normalize per-channel then optional gamma."""
    out = img_rgb.astype(np.float32)
    for c in range(3):
        ch = out[..., c]
        lo, hi = float(ch.min()), float(ch.max())
        if hi - lo < 1e-6:
            continue
        out[..., c] = (ch - lo) / (hi - lo) * 255.0
    if gamma and abs(gamma - 1.0) > 1e-3:
        out = (np.clip(out / 255.0, 0, 1) ** (1.0 / float(gamma))) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def preprocess_rgb(
    img_rgb: np.ndarray,
    *,
    use_clahe: bool = False,
    clahe_clip: float = 2.0,
    clahe_grid: int = 8,
    use_normalize: bool = False,
    gamma: float = 1.0,
) -> np.ndarray:
    """Apply selected preprocessing in a deterministic order: CLAHE → normalize+gamma."""
    out = img_rgb
    if use_clahe:
        out = apply_clahe(out, clip=clahe_clip, grid=clahe_grid)
    if use_normalize:
        out = apply_normalize(out, gamma=gamma)
    return out
