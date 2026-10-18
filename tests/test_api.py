import io

import pytest
from PIL import Image

from tests.conftest import png_bytes

COMBOS = ["hog_svm/opencv", "faster_rcnn/pytorch", "retinanet/pytorch", "retinanet/tensorflow"]


def post(client, url, data=None, **kw):
    return client.post(url, files={"file": ("a.png", data or png_bytes(), "image/png")}, **kw)


def test_health_matrix(client):
    j = client.get("/health").json()
    assert j["status"] == "ok"
    assert set(j["backends"]) == set(COMBOS)
    assert j["algorithms"] == {"hog_svm": ["opencv"], "faster_rcnn": ["pytorch"], "retinanet": ["pytorch", "tensorflow"]}
    assert "faster_rcnn/tensorflow" in j["unavailable"]
    keys = {"framework", "algorithm", "loaded", "error", "model", "parameters", "model_size_mb", "device", "mock", "via"}
    assert keys <= set(j["backends"][COMBOS[0]])


def test_predict_defaults_and_fields(client):
    j = post(client, "/predict", params={"threshold": 0.5}).json()
    assert (j["algorithm"], j["framework"]) == ("retinanet", "pytorch")
    assert j["mock"] is True and j["model"].startswith("mock-")
    assert j["score_meaning"] == "probability-like"
    assert j["settings"]["nms_iou"] == 0.5 and "nms_applied" in j["settings"]
    assert j["model_info"]["classes_supported"] == 80
    assert j["image"] == {"width": 320, "height": 240}
    assert j["detections"]
    for d in j["detections"]:
        x1, y1, x2, y2 = d["box"]
        assert 0 <= x1 < x2 <= 320 and 0 <= y1 < y2 <= 240 and d["confidence"] >= 0.5


@pytest.mark.parametrize("combo", COMBOS)
def test_every_valid_combo(client, combo):
    a, f = combo.split("/")
    r = post(client, "/predict", params={"algorithm": a, "framework": f})
    assert r.status_code == 200 and r.json()["framework"] == f


def test_hog_defaults_to_opencv(client):
    j = post(client, "/predict", params={"algorithm": "hog_svm"}).json()
    assert j["framework"] == "opencv" and j["score_meaning"] == "rescaled SVM score"
    assert {"hog_stride", "hog_scale", "hit_threshold"} <= set(j["settings"])
    assert j["model_info"]["parameters"] is None and j["model_info"]["classes_supported"] == 1
    assert {d["label"] for d in j["detections"]} == {"person"}


@pytest.mark.parametrize("params", [
    {"algorithm": "hog_svm", "framework": "pytorch"},
    {"algorithm": "faster_rcnn", "framework": "tensorflow"},
    {"algorithm": "yolo"},
    {"algorithm": "retinanet", "framework": "jax"},
])
def test_invalid_combo_lists_valid_ones(client, params):
    r = post(client, "/predict", params=params)
    assert r.status_code == 422
    assert "retinanet/tensorflow" in r.json()["detail"]


@pytest.mark.parametrize("params", [
    {"threshold": 1.5}, {"nms_iou": 0.05}, {"nms_iou": 1.5},
    {"hog_stride": 5}, {"hog_scale": 1.0}, {"hog_scale": 2}, {"hit_threshold": 3},
])
def test_bad_parameters_422(client, params):
    assert post(client, "/predict", params=params).status_code == 422


def test_error_codes(client, monkeypatch):
    assert client.post("/predict", files={"file": ("a.txt", b"hello", "text/plain")}).status_code == 400
    assert client.post("/predict", files={"file": ("a.png", b"", "image/png")}).status_code == 400
    assert post(client, "/predict", data=png_bytes()[:60]).status_code == 400  # truncated
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.0001")
    assert post(client, "/predict").status_code == 413


def test_exif_rotation_and_alpha(client):
    img = Image.new("RGBA", (200, 100), (255, 0, 0, 128))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees
    buf = io.BytesIO()
    img.save(buf, "PNG", exif=exif)
    j = post(client, "/predict", data=buf.getvalue()).json()
    assert j["image"] == {"width": 100, "height": 200}


def test_compare_all_and_subset(client):
    j = post(client, "/compare", params={"threshold": 0.05}).json()
    assert set(j["results"]) == set(COMBOS) and j["threshold"] == 0.05
    assert all(r["ok"] for r in j["results"].values())
    j = post(client, "/compare", params={"combos": "hog_svm/opencv,retinanet/pytorch"}).json()
    assert set(j["results"]) == {"hog_svm/opencv", "retinanet/pytorch"}


def test_compare_known_unavailable_does_not_fail(client):
    j = post(client, "/compare", params={"combos": "faster_rcnn/tensorflow,retinanet/pytorch"}).json()
    assert j["results"]["faster_rcnn/tensorflow"]["unavailable"] is True
    assert j["results"]["faster_rcnn/tensorflow"]["ok"] is False
    assert j["results"]["retinanet/pytorch"]["ok"] is True


def test_compare_unknown_names_422(client):
    assert post(client, "/compare", params={"combos": "yolo/pytorch"}).status_code == 422
    assert post(client, "/compare", params={"combos": "retinanet/jax"}).status_code == 422


def test_threshold_filters(client):
    low = post(client, "/predict", params={"threshold": 0.05}).json()["detections"]
    high = post(client, "/predict", params={"threshold": 0.9}).json()["detections"]
    assert len(low) > len(high) > 0


def test_nms_knob_mock(client):
    on = post(client, "/predict", params={"threshold": 0.05, "nms_iou": 0.5}).json()["detections"]
    off = post(client, "/predict", params={"threshold": 0.05, "nms_iou": 1.0}).json()["detections"]
    assert len(off) > len(on)


def test_samples_listing_and_traversal(client, tmp_path, monkeypatch):
    from app import config

    root = tmp_path / "samples"
    root.mkdir()
    (root / "a.jpg").write_bytes(png_bytes())
    (root / "notes.txt").write_text("x")
    (tmp_path / "secret.jpg").write_bytes(b"secret")
    monkeypatch.setattr(config, "SAMPLES_DIR", root)
    assert client.get("/samples").json() == {"samples": ["a.jpg"]}
    assert client.get("/samples/a.jpg").status_code == 200
    assert client.get("/samples/notes.txt").status_code == 404
    for bad in ["../secret.jpg", "..%2Fsecret.jpg", "%2e%2e/secret.jpg", "..%5Csecret.jpg",
                "%2Fetc%2Fpasswd", "C:%5Cwindows%5Cwin.ini", "%2e%2e%2fsecret.jpg"]:
        r = client.get(f"/samples/{bad}")
        assert r.status_code in (404, 400, 422) and b"secret" not in r.content, bad


def test_index_and_static(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/demo/app.js").status_code == 200


def test_there_is_no_endpoint_that_writes_photos(client, tmp_path, monkeypatch):
    from app import config

    monkeypatch.setattr(config, "SAMPLES_DIR", tmp_path / "samples")
    for method in ("post", "put", "delete"):
        r = getattr(client, method)("/samples", files={"file": ("a.png", png_bytes(), "image/png")}) if method != "delete" else client.delete("/samples")
        assert r.status_code in (404, 405)
    assert not (tmp_path / "samples").exists()
