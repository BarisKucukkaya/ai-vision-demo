import types

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from app.adapters.torchvision_detectors import FasterRCNNPyTorch, RetinaNetPyTorch  # noqa: E402
from app.schemas import ImageInput, Options  # noqa: E402


@pytest.fixture(autouse=True)
def random_weights(monkeypatch):
    monkeypatch.setenv("DEMO_RANDOM_WEIGHTS", "1")
    monkeypatch.setenv("DEMO_DEVICE", "cpu")


def image(w=200, h=150):
    rng = np.random.default_rng(0)
    return ImageInput(image=Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)))


@pytest.mark.parametrize("cls,algorithm", [(RetinaNetPyTorch, "retinanet"), (FasterRCNNPyTorch, "faster_rcnn")])
def test_random_weights_end_to_end(cls, algorithm):
    adapter = cls()
    pred = adapter.predict(image(), 0.0)
    d = pred.to_dict()
    assert d["algorithm"] == algorithm and d["framework"] == "pytorch"
    assert d["mock"] is True and "RANDOM weights - test only" in d["model"]
    assert d["model_info"]["parameters"] > 30_000_000 and d["model_info"]["classes_supported"] == 80
    assert d["device"] == "cpu" and d["settings"]["nms_applied"] is True
    for det in pred.detections:
        x1, y1, x2, y2 = det.box
        assert 0 <= x1 < x2 <= 200 and 0 <= y1 < y2 <= 150
    assert adapter.info()["loaded"] is True


def test_weights_enums_have_coco_v1():
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, RetinaNet_ResNet50_FPN_Weights

    assert RetinaNet_ResNet50_FPN_Weights.COCO_V1 and FasterRCNN_ResNet50_FPN_Weights.COCO_V1


class FakeDetector:
    def __init__(self):
        self.nms_thresh = 0.5
        self.roi_heads = types.SimpleNamespace(nms_thresh=0.5)
        self.seen = []

    def __call__(self, batch):
        self.seen.append((self.nms_thresh, self.roi_heads.nms_thresh))
        return [{"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.int64)}]


@pytest.mark.parametrize("cls,idx", [(RetinaNetPyTorch, 0), (FasterRCNNPyTorch, 1)])
def test_nms_applied_during_call_and_restored(cls, idx):
    adapter = cls()
    adapter._torch, adapter._model, adapter._loaded = torch, FakeDetector(), True
    adapter.predict(image(), 0.5, Options(nms_iou=0.2))
    adapter.predict(image(), 0.5, Options(nms_iou=0.9))
    adapter.predict(image(), 0.5)
    assert [s[idx] for s in adapter._model.seen] == [0.2, 0.9, 0.5]  # no leaking between requests
    assert adapter._model.nms_thresh == 0.5 and adapter._model.roi_heads.nms_thresh == 0.5


def test_unknown_input_does_not_break_api(monkeypatch):
    from tests.conftest import make_client, png_bytes

    with make_client(monkeypatch, DEMO_MOCK="0", DEMO_COMBOS="retinanet/pytorch,faster_rcnn/pytorch") as c:
        files = {"file": ("a.png", png_bytes(), "image/png")}
        r = c.post("/predict", params={"algorithm": "faster_rcnn", "threshold": 0.05}, files=files)
        assert r.status_code == 200 and r.json()["mock"] is True
        assert c.get("/health").json()["backends"]["faster_rcnn/pytorch"]["loaded"] is True
