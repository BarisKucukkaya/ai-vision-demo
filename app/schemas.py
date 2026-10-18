"""Plain dataclasses shared by the API, the adapters and the post-processing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass
class ImageInput:
    """A decoded upload: EXIF-rotated RGB PIL image plus the original bytes (for workers)."""

    image: Image.Image
    raw: bytes = b""
    filename: str = "upload"

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height


@dataclass
class Detection:
    label: str
    confidence: float
    box: list[float]  # [x1, y1, x2, y2] in pixels of the original image

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "confidence": self.confidence, "box": self.box}


@dataclass
class Options:
    """Per-request settings."""

    nms_iou: float = 0.5
    hog_stride: int = 8
    hog_scale: float = 1.05
    hit_threshold: float = 0.0


@dataclass
class RawOutput:
    """What an adapter's _predict returns: boxes in original pixels, xyxy."""

    boxes: Any  # array-like (N, 4)
    scores: Any  # array-like (N,)
    labels: Any  # array-like (N,) of int class ids
    settings: dict[str, Any] = field(default_factory=dict)
    inference_ms: float | None = None  # set by adapters that time the model call themselves


@dataclass
class Prediction:
    algorithm: str
    framework: str
    model: str
    width: int
    height: int
    detections: list[Detection]
    inference_ms: float
    via: str = "local"
    device: str = "cpu"
    mock: bool = False
    score_meaning: str = "probability-like"
    settings: dict[str, Any] = field(default_factory=dict)
    model_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "framework": self.framework,
            "model": self.model,
            "image": {"width": self.width, "height": self.height},
            "detections": [d.to_dict() for d in self.detections],
            "inference_ms": round(self.inference_ms, 1),
            "via": self.via,
            "device": self.device,
            "mock": self.mock,
            "score_meaning": self.score_meaning,
            "settings": self.settings,
            "model_info": self.model_info,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Prediction":
        return cls(
            algorithm=d["algorithm"],
            framework=d["framework"],
            model=d["model"],
            width=d["image"]["width"],
            height=d["image"]["height"],
            detections=[Detection(x["label"], x["confidence"], list(x["box"])) for x in d["detections"]],
            inference_ms=d["inference_ms"],
            via=d.get("via", "local"),
            device=d.get("device", "cpu"),
            mock=d.get("mock", False),
            score_meaning=d.get("score_meaning", "probability-like"),
            settings=d.get("settings", {}),
            model_info=d.get("model_info", {}),
        )
