"""Open-vocabulary 2D detection via ultralytics YOLO-World.

Deliberately lazy: the baseline prototype must still import even when
`ultralytics` / `torch` aren't installed. Only when the user triggers a
detector run do we import and load the model.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


# Corporate-network friendly: point Python at certifi's bundle so the CLIP text
# encoder (pulled by YOLO-World on first `set_classes`) can download.
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass


_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, object] = {}
_LAST_CLASSES: list[str] = []


def _target_device() -> str:
    """Pick GPU if available. Kept as a function (not a constant) so that
    `CUDA_VISIBLE_DEVICES` changes at runtime are honored on model reload."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

# Where to find the YOLO weight file. Bundled with the repo in `assets/` so
# the app works out-of-the-box on an air-gapped machine; falls through to
# ultralytics' own download path if the file is missing.
_DEFAULT_WEIGHT = Path(__file__).resolve().parent.parent / "assets" / "yolov8s-world.pt"


@dataclass
class Detection2D:
    class_name: str
    confidence: float
    bbox: np.ndarray                  # (4,) float32: x1, y1, x2, y2
    mask: Optional[np.ndarray] = None # (H, W) bool, optional
    extra: dict = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def is_available() -> bool:
    """True if `ultralytics` is installed; False otherwise."""
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:
        return False


def load_model(weights: Optional[str] = None):
    """Load (and cache) a YOLO-World model. Thread-safe."""
    from ultralytics import YOLOWorld
    key = weights or str(_DEFAULT_WEIGHT)
    with _MODEL_LOCK:
        if key not in _MODEL_CACHE:
            path = Path(key)
            if path.exists():
                model = YOLOWorld(str(path))
            else:
                # Let ultralytics resolve / auto-download (e.g. 'yolov8s-world.pt')
                model = YOLOWorld(key)
            # Move to GPU once at load time. `set_classes` will re-encode text
            # on whatever device the CLIP model lives on, so we also force
            # re-placement after each call (see `detect`).
            try:
                model.to(_target_device())
            except Exception:
                pass
            _MODEL_CACHE[key] = model
        return _MODEL_CACHE[key]


def _move_prompt_embeddings_to_device(model, device: str) -> None:
    """After `YOLOWorld.set_classes`, the new class text embeddings can end up
    on CPU while the detection head is on CUDA, causing::

        RuntimeError: Expected all tensors to be on the same device, but got
        index is on cpu, different from other tensors on cuda:0

    This is a known ultralytics quirk with cached models. We brute-force it
    by moving the whole underlying nn.Module to the target device — any
    fresh CPU tensors inside get caught.
    """
    try:
        model.to(device)
    except Exception:
        pass
    # Belt-and-suspenders: ultralytics keeps text features on the detection
    # head as `txt_feats` in some versions. Move that explicitly.
    try:
        import torch
        inner = getattr(model, "model", None)
        if inner is not None:
            for name in ("txt_feats", "text_feats", "class_embeddings"):
                feat = getattr(inner, name, None)
                if isinstance(feat, torch.Tensor):
                    setattr(inner, name, feat.to(device))
    except Exception:
        pass


def detect(
    image_rgb: np.ndarray,
    prompts: Sequence[str],
    *,
    conf: float = 0.05,
    iou: float = 0.45,
    weights: Optional[str] = None,
    max_det: int = 50,
) -> list[Detection2D]:
    """Run YOLO-World on an RGB image with open-vocab text prompts.

    Args:
        image_rgb: (H, W, 3) uint8 RGB.
        prompts: list of class name strings, e.g. ['sphere', 'bolt'].
        conf: confidence threshold.
        iou: NMS IoU threshold.
        weights: optional path to a YOLO-World weight file.

    Returns a list of Detection2D, sorted by descending confidence.
    """
    if not is_available():
        raise RuntimeError(
            "ultralytics not installed — `pip install ultralytics` inside the "
            "loc-proto env, or remove this optional dep from requirements.txt."
        )
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    prompts = [p.strip() for p in prompts if p and p.strip()]
    if not prompts:
        return []

    model = load_model(weights)
    device = _target_device()
    # set_classes is a CLIP text-encoder forward pass; skip if prompts unchanged.
    # On re-entry with new prompts the fresh embeddings can land on CPU even
    # though the YOLO backbone is on CUDA, so we re-place the whole model.
    global _LAST_CLASSES
    with _MODEL_LOCK:
        if prompts != _LAST_CLASSES:
            model.set_classes(list(prompts))
            _move_prompt_embeddings_to_device(model, device)
            _LAST_CLASSES = list(prompts)

    result = model.predict(
        image_rgb, conf=float(conf), iou=float(iou),
        max_det=int(max_det), verbose=False, device=device,
    )[0]
    out: list[Detection2D] = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return out
    xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
    confs = boxes.conf.cpu().numpy().astype(np.float32)
    clsids = boxes.cls.cpu().numpy().astype(int)
    names = [prompts[i] if 0 <= i < len(prompts) else f"class_{i}" for i in clsids]
    for i in range(len(xyxy)):
        out.append(Detection2D(
            class_name=names[i],
            confidence=float(confs[i]),
            bbox=xyxy[i],
        ))
    out.sort(key=lambda d: d.confidence, reverse=True)
    return out
