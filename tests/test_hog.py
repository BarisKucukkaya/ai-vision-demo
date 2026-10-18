import sys
import time

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("cv2")
from app.adapters import base  # noqa: E402
from app.adapters.base import BackendUnavailable  # noqa: E402
from app.adapters.hog_opencv import HogOpenCV  # noqa: E402
from app.schemas import ImageInput, Options  # noqa: E402
from tests.conftest import make_client, png_bytes  # noqa: E402


def inp(img):
    return ImageInput(image=img.convert("RGB"))


def noise(w, h, seed):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def test_blank_image_returns_empty_list():
    pred = HogOpenCV().predict(inp(Image.new("RGB", (320, 240), (200, 200, 200))), 0.05)
    assert pred.detections == [] and pred.algorithm == "hog_svm" and pred.model_info["parameters"] is None


def test_noise_image_boxes_inside_and_person_only():
    pred = HogOpenCV().predict(inp(noise(640, 480, 0)), 0.05, Options(hit_threshold=0.0, hog_stride=16))
    for d in pred.detections:
        x1, y1, x2, y2 = d.box
        assert d.label == "person" and 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480
        assert 0.05 <= d.confidence <= 1


def test_nms_off_returns_at_least_as_many_and_settings_restored():
    a = HogOpenCV()
    img = inp(noise(300, 300, 1))
    on = a.predict(img, 0.0, Options(nms_iou=0.3, hog_stride=8))
    off = a.predict(img, 0.0, Options(nms_iou=1.0, hog_stride=8))
    again = a.predict(img, 0.0, Options(nms_iou=0.3, hog_stride=8))
    assert len(off.detections) >= len(on.detections)
    assert len(again.detections) == len(on.detections)  # nothing leaked between requests
    assert off.settings["nms_applied"] is False and on.settings["nms_applied"] is True


def test_large_image_is_downscaled_and_mapped_back():
    pred = HogOpenCV().predict(inp(Image.new("RGB", (2400, 1200), (10, 10, 10))), 0.05)
    assert (pred.width, pred.height) == (2400, 1200)


def test_missing_hogdescriptor_gives_pin_hint(monkeypatch):
    import cv2

    monkeypatch.delattr(cv2, "HOGDescriptor")
    with pytest.raises(BackendUnavailable, match=r"opencv-python-headless>=4.8,<5"):
        HogOpenCV().ensure_loaded()


def test_missing_opencv_gives_pin_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(BackendUnavailable, match=r"<5"):
        HogOpenCV().ensure_loaded()


def test_error_is_cached_then_retried(monkeypatch):
    a = HogOpenCV()
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "cv2", None)
        with pytest.raises(BackendUnavailable):
            a.ensure_loaded()
    with pytest.raises(BackendUnavailable):  # still cached: no retry yet
        a.ensure_loaded()
    a._error_time = time.time() - base.Adapter.retry_after_error - 1
    a.ensure_loaded()  # retried after 30 s and works now


def test_api_hog_real(monkeypatch):
    with make_client(monkeypatch, DEMO_MOCK="0", DEMO_COMBOS="hog_svm/opencv") as c:
        files = {"file": ("a.png", png_bytes(), "image/png")}
        r = c.post("/predict", params={"algorithm": "hog_svm"}, files=files)
        assert r.status_code == 200 and r.json()["framework"] == "opencv" and r.json()["mock"] is False
        r = c.post("/predict", params={"algorithm": "hog_svm", "framework": "pytorch"}, files=files)
        assert r.status_code == 422
        r = c.post("/predict", params={"algorithm": "retinanet"}, files=files)
        assert r.status_code == 422  # valid combo, but not served by this process
