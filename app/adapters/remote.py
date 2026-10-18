"""Forwards a combo to a worker process over HTTP (distributed mode)."""
from __future__ import annotations

import time
from typing import Any

import httpx

from .. import config
from ..schemas import ImageInput, Options, Prediction
from .base import Adapter, BackendError, BackendUnavailable

INFO_CACHE_S = 3.0


class RemoteAdapter(Adapter):
    def __init__(self, algorithm: str, framework: str, url: str) -> None:
        super().__init__()
        self.algorithm, self.framework, self.url = algorithm, framework, url.rstrip("/")
        self.combo = f"{algorithm}/{framework}"
        self._info: dict[str, Any] | None = None
        self._info_time = 0.0

    def ensure_loaded(self) -> None:  # the worker loads its own models
        pass

    def predict(self, image_input: ImageInput, threshold: float, options: Options | None = None) -> Prediction:
        options = options or Options()
        params = {
            "algorithm": self.algorithm, "framework": self.framework, "threshold": threshold,
            "nms_iou": options.nms_iou, "hog_stride": options.hog_stride,
            "hog_scale": options.hog_scale, "hit_threshold": options.hit_threshold,
        }
        try:
            resp = httpx.post(
                f"{self.url}/predict", params=params,
                files={"file": (image_input.filename, image_input.raw, "application/octet-stream")},
                timeout=config.remote_timeout_s(),
            )
        except httpx.HTTPError as exc:
            raise BackendUnavailable(f"Worker at {self.url} is not reachable ({type(exc).__name__}).") from exc
        if resp.status_code == 503:
            raise BackendUnavailable(_detail(resp))
        if resp.status_code != 200:
            raise BackendError(f"Worker returned {resp.status_code}: {_detail(resp)}")
        pred = Prediction.from_dict(resp.json())
        pred.via = "remote"
        self._info_time = 0.0  # refresh model info (parameters) after the first successful prediction
        self.parameters = pred.model_info.get("parameters")
        return pred

    def _fetch_info(self) -> dict[str, Any]:
        now = time.time()
        if self._info is not None and now - self._info_time < INFO_CACHE_S:
            return self._info
        try:
            data = httpx.get(f"{self.url}/health", timeout=3.0).json()["backends"][self.combo]
            data = dict(data, via="remote")
        except Exception as exc:
            data = {
                "framework": self.framework, "algorithm": self.algorithm, "loaded": False,
                "error": f"Worker at {self.url} is not reachable ({type(exc).__name__}).",
                "model": None, "parameters": None, "model_size_mb": None, "device": None,
                "mock": False, "via": "remote",
            }
        self._info, self._info_time = data, now
        return data

    def info(self) -> dict[str, Any]:
        return self._fetch_info()


def _detail(resp: httpx.Response) -> str:
    try:
        return str(resp.json().get("detail", resp.text))
    except Exception:
        return resp.text
