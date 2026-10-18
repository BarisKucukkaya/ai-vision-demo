"""Environment variables. Read at call time so tests can change them."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = Path(os.environ.get("DEMO_SAMPLES_DIR") or ROOT / "samples")
STATIC_DIR = Path(__file__).resolve().parent / "static"

ALL_COMBOS = [
    "hog_svm/opencv",
    "faster_rcnn/pytorch",
    "retinanet/pytorch",
    "retinanet/tensorflow",
]


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def demo_combos() -> list[str]:
    raw = os.environ.get("DEMO_COMBOS", "").strip()
    if not raw:
        return list(ALL_COMBOS)
    return [c.strip() for c in raw.split(",") if c.strip()]


def demo_remotes() -> dict[str, str]:
    raw = os.environ.get("DEMO_REMOTES", "").strip()
    remotes: dict[str, str] = {}
    for part in raw.split(","):
        if "=" in part:
            combo, url = part.split("=", 1)
            remotes[combo.strip()] = url.strip().rstrip("/")
    return remotes


def mock_mode() -> bool:
    return _flag("DEMO_MOCK")


def random_weights() -> bool:
    return _flag("DEMO_RANDOM_WEIGHTS")


def preload() -> bool:
    return _flag("DEMO_PRELOAD", "1")


def device_choice() -> str:
    return os.environ.get("DEMO_DEVICE", "auto").strip().lower()


def tf_preset() -> str:
    return os.environ.get("TF_PRESET", "retinanet_resnet50_fpn_coco")


def max_upload_mb() -> float:
    return float(os.environ.get("MAX_UPLOAD_MB", "10"))


def remote_timeout_s() -> float:
    return float(os.environ.get("REMOTE_TIMEOUT_S", "120"))

