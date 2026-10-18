import os
import socket
import subprocess
import sys
import time

import httpx
import pytest

from tests.conftest import ROOT, make_client, png_bytes


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_worker(port, combos, extra_env=None):
    env = dict(os.environ, DEMO_MOCK="1", DEMO_PRELOAD="0", DEMO_COMBOS=combos, DEMO_REMOTES="")
    env.update(extra_env or {})
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port), "--log-level", "warning"],
        cwd=ROOT, env=env,
    )
    for _ in range(100):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("worker did not start")


@pytest.fixture
def worker():
    port = free_port()
    proc = start_worker(port, "retinanet/pytorch,faster_rcnn/pytorch")
    yield port, proc
    proc.kill()
    proc.wait()


def files():
    return {"file": ("a.png", png_bytes(), "image/png")}


def test_prediction_through_gateway_is_remote(monkeypatch, worker):
    port, _ = worker
    url = f"http://127.0.0.1:{port}"
    remotes = f"retinanet/pytorch={url},faster_rcnn/pytorch={url}"
    with make_client(monkeypatch, DEMO_COMBOS="hog_svm/opencv", DEMO_REMOTES=remotes) as c:
        j = c.post("/predict", params={"algorithm": "retinanet"}, files=files()).json()
        assert j["via"] == "remote" and j["model"].startswith("mock-") and j["detections"]
        assert j["model_info"]["parameters"]
        local = c.post("/predict", params={"algorithm": "hog_svm"}, files=files()).json()
        assert local["via"] == "local"
        health = c.get("/health").json()["backends"]
        assert health["retinanet/pytorch"]["via"] == "remote"
        assert set(health) == {"hog_svm/opencv", "retinanet/pytorch", "faster_rcnn/pytorch"}
        # an algorithm served by nobody is simply not registered on this gateway
        assert c.post("/predict", params={"algorithm": "retinanet", "framework": "tensorflow"}, files=files()).status_code == 422


def test_nms_setting_travels_to_worker(monkeypatch, worker):
    port, _ = worker
    with make_client(monkeypatch, DEMO_COMBOS="hog_svm/opencv", DEMO_REMOTES=f"retinanet/pytorch=http://127.0.0.1:{port}") as c:
        on = c.post("/predict", params={"threshold": 0.05, "nms_iou": 0.5}, files=files()).json()
        off = c.post("/predict", params={"threshold": 0.05, "nms_iou": 1.0}, files=files()).json()
        assert len(off["detections"]) > len(on["detections"]) and off["settings"]["nms_iou"] == 1.0


def test_worker_down_gives_503(monkeypatch):
    dead = f"http://127.0.0.1:{free_port()}"
    with make_client(monkeypatch, DEMO_COMBOS="hog_svm/opencv", DEMO_REMOTES=f"retinanet/pytorch={dead}") as c:
        r = c.post("/predict", params={"algorithm": "retinanet"}, files=files())
        assert r.status_code == 503 and "not reachable" in r.json()["detail"]
        j = c.post("/compare", params={"combos": "retinanet/pytorch,hog_svm/opencv"}, files=files()).json()
        assert j["results"]["retinanet/pytorch"]["unavailable"] is True and j["results"]["hog_svm/opencv"]["ok"]
        info = c.get("/health").json()["backends"]["retinanet/pytorch"]
        assert info["loaded"] is False and info["error"]


def test_worker_503_is_passed_on(monkeypatch):
    """The worker cannot import cv2 (blocked via sitecustomize), so it answers 503 with the install hint."""
    port = free_port()
    proc = start_worker(port, "hog_svm/opencv", {"DEMO_MOCK": "0", "PYTHONPATH": str(ROOT / "tests" / "blockers")})
    try:
        with make_client(monkeypatch, DEMO_COMBOS="retinanet/pytorch", DEMO_REMOTES=f"hog_svm/opencv=http://127.0.0.1:{port}") as c:
            r = c.post("/predict", params={"algorithm": "hog_svm"}, files=files())
            assert r.status_code == 503 and "opencv-python-headless>=4.8,<5" in r.json()["detail"]
    finally:
        proc.kill()
        proc.wait()


def test_parameters_filled_after_first_prediction_even_if_worker_was_loading(monkeypatch):
    from app.adapters.remote import RemoteAdapter

    port = free_port()
    adapter = RemoteAdapter("retinanet", "pytorch", f"http://127.0.0.1:{port}")
    assert adapter.info()["parameters"] is None  # worker not up yet: cached error info
    proc = start_worker(port, "retinanet/pytorch")
    try:
        from app.schemas import ImageInput
        from PIL import Image

        adapter.predict(ImageInput(image=Image.new("RGB", (64, 64)), raw=png_bytes(64, 64)), 0.5)
        assert adapter.info()["parameters"]  # refreshed, not stuck on the early None
    finally:
        proc.kill()
        proc.wait()
