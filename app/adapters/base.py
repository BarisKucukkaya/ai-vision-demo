"""Adapter base class: one combo (<algorithm>/<framework>) behind a common interface."""
from __future__ import annotations

import threading
import time
from typing import Any

from ..coco_classes import COCO_CATEGORIES
from ..postprocess import build_detections
from ..schemas import ImageInput, Options, Prediction, RawOutput


class BackendUnavailable(Exception):
    """Library missing, weights unreachable, worker down (HTTP 503)."""


class BackendError(Exception):
    """Anything else that goes wrong while predicting (HTTP 500)."""


# excerpt: backend_contract
class Adapter:
    algorithm = ""
    framework = ""
    model_name = ""
    score_meaning = "probability-like"
    classes_supported = len(COCO_CATEGORIES)
    retry_after_error = 30.0  # seconds before a failed load is tried again

    def __init__(self) -> None:
        self.parameters: int | None = None
        self.model_size_mb: float | None = None
        self.device = "cpu"
        self.mock = False
        self._loaded = False
        self._error: str | None = None
        self._error_time = 0.0
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()  # one inference at a time

    def _load(self) -> None:
        raise NotImplementedError

    def _predict(self, image_input: ImageInput, options: Options) -> RawOutput:
        raise NotImplementedError

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._load_lock:  # lazy, thread-safe, loads once
            if self._loaded:
                return
            if self._error and time.time() - self._error_time < self.retry_after_error:
                raise BackendUnavailable(self._error)
            try:
                self._load()
            except BackendUnavailable as exc:
                self._error, self._error_time = str(exc), time.time()
                raise
            except Exception as exc:  # weights unreachable, bad install, ...
                self._error, self._error_time = f"{type(exc).__name__}: {exc}", time.time()
                raise BackendUnavailable(self._error) from exc
            self._error = None
            self._loaded = True

    def predict(self, image_input: ImageInput, threshold: float, options: Options | None = None) -> Prediction:
        options = options or Options()
        self.ensure_loaded()
        with self._infer_lock:
            start = time.perf_counter()
            try:
                raw = self._predict(image_input, options)
            except (BackendUnavailable, BackendError):
                raise
            except Exception as exc:
                raise BackendError(f"{type(exc).__name__}: {exc}") from exc
            inference_ms = (time.perf_counter() - start) * 1000.0
        # endexcerpt
        if raw.inference_ms is not None:
            inference_ms = raw.inference_ms
        detections = build_detections(
            raw.boxes, raw.scores, raw.labels, COCO_CATEGORIES, threshold,
            image_input.width, image_input.height,
        )
        return Prediction(
            algorithm=self.algorithm,
            framework=self.framework,
            model=self.model_name,
            width=image_input.width,
            height=image_input.height,
            detections=detections,
            inference_ms=inference_ms,
            via="local",
            device=self.device,
            mock=self.mock,
            score_meaning=self.score_meaning,
            settings=raw.settings,
            model_info=self.model_info(),
        )

    def model_info(self) -> dict[str, Any]:
        return {
            "parameters": self.parameters,
            "model_size_mb": self.model_size_mb,
            "classes_supported": self.classes_supported,
        }

    def info(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "algorithm": self.algorithm,
            "loaded": self._loaded,
            "error": None if self._loaded else self._error,
            "model": self.model_name,
            "parameters": self.parameters,
            "model_size_mb": self.model_size_mb,
            "device": self.device,
            "mock": self.mock,
            "via": "local",
        }
