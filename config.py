"""Shared constants for the localization prototype."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
CACHE_DIR = ROOT / "cache"
UPLOAD_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

DISPLAY_MAX_POINTS = 50_000
DISPLAY_MODEL_MAX_POINTS = 8_000

DEFAULT_VOXEL_SCENE_MM = 3.0
DEFAULT_VOXEL_MODEL_MM = 2.0

MM_PER_UNIT_CHOICES = {
    "auto": None,
    "millimeters (x1)": 1.0,
    "centimeters (x10)": 10.0,
    "meters (x1000)": 1000.0,
    "inches (x25.4)": 25.4,
}
