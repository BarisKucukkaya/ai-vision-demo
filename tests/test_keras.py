import numpy as np
import pytest
from PIL import Image

pytest.importorskip("tensorflow")
pytest.importorskip("keras_hub")
from app.adapters.keras_retinanet import KerasRetinaNet, letterbox, to_original  # noqa: E402
from app.schemas import ImageInput, Options  # noqa: E402


@pytest.fixture(autouse=True)
def random_weights(monkeypatch):
    monkeypatch.setenv("DEMO_RANDOM_WEIGHTS", "1")


def test_letterbox_worked_example():
    canvas, scale, ox, oy = letterbox(Image.new("RGB", (400, 200)), 800, 800)
    assert canvas.shape == (800, 800, 3)
    assert (scale, ox, oy) == (2.0, 0, 200)
    yxyx = np.array([[300, 200, 500, 600]], dtype=float)
    xyxy = yxyx[:, [1, 0, 3, 2]]
    assert xyxy.tolist() == [[200, 300, 600, 500]]
    assert to_original(xyxy, scale, ox, oy).tolist() == [[100, 50, 300, 150]]


def test_letterbox_portrait_and_content_placement():
    img = Image.new("RGB", (100, 400), (255, 255, 255))
    canvas, scale, ox, oy = letterbox(img, 800, 800)
    assert scale == 2.0 and ox == 300 and oy == 0
    assert canvas[400, 400].tolist() == [255, 255, 255] and canvas[400, 10].tolist() == [0, 0, 0]


class FakeModel:
    """Returns one box in yxyx model pixels, like KerasHub does."""

    bounding_box_format = "yxyx"

    def __init__(self):
        self.prediction_decoder = type("D", (), {"iou_threshold": 0.5, "confidence_threshold": 0.05})()
        self.predict_function = object()
        self.seen = []

    def predict(self, batch, verbose=0):
        assert batch.shape == (1, 800, 800, 3) and batch.dtype == np.float32
        self.seen.append(self.prediction_decoder.iou_threshold)
        return {
            "boxes": np.array([[[300, 200, 500, 600], [0, 0, 0, 0]]], dtype="float32"),
            "confidence": np.array([[0.9, 0.0]], dtype="float32"),
            "labels": np.array([[3, 0]]),
            "num_detections": np.array([1]),
        }


def test_fake_model_box_mapping_to_original_pixels():
    import keras

    adapter = KerasRetinaNet()
    adapter._keras, adapter._model, adapter._loaded = keras, FakeModel(), True
    adapter.input_hw = (800, 800)
    image = ImageInput(image=Image.new("RGB", (400, 200)))
    pred = adapter.predict(image, 0.5, Options(nms_iou=0.3))
    assert [(d.label, d.box) for d in pred.detections] == [("car", [100.0, 50.0, 300.0, 150.0])]
    assert adapter._model.seen == [0.3, 0.3, 0.5]  # retrace run, timed run, retrace back
    assert pred.inference_ms >= 0
    assert adapter._model.prediction_decoder.iou_threshold == 0.5  # restored


def test_random_weights_end_to_end():
    adapter = KerasRetinaNet()
    img = ImageInput(image=Image.fromarray(np.random.default_rng(0).integers(0, 255, (120, 200, 3), dtype=np.uint8)))
    pred = adapter.predict(img, 0.0)
    d = pred.to_dict()
    assert d["framework"] == "tensorflow" and d["mock"] is True and "RANDOM weights" in d["model"]
    assert d["model_info"]["parameters"] > 1_000_000 and d["model_info"]["classes_supported"] == 80
    for det in pred.detections:
        x1, y1, x2, y2 = det.box
        assert 0 <= x1 < x2 <= 200 and 0 <= y1 < y2 <= 120
    assert adapter.input_hw == (256, 256)


def test_keras_traces_decoder_iou_so_adapter_must_retrace():
    """Proves the real behaviour: changing the decoder IoU after the first predict() has no effect on the
    compiled predict function; dropping it applies the new value. The adapter relies on this."""
    import keras
    from keras_hub.src.layers.modeling.non_max_supression import NonMaxSuppression

    dec = NonMaxSuppression(bounding_box_format="yxyx", from_logits=False, iou_threshold=0.5,
                            confidence_threshold=0.05, max_detections=10)
    b, s = keras.Input(batch_shape=(1, 3, 4)), keras.Input(batch_shape=(1, 3, 2))
    model = keras.Model([b, s], dec(b, s)["num_detections"])
    boxes = np.array([[[0, 0, 10, 10], [0, 0, 10, 10.5], [50, 50, 60, 60]]], "float32")
    scores = np.array([[[.9, .1], [.8, .1], [.7, .1]]], "float32")

    def count():
        return int(model.predict([boxes, scores], verbose=0)[0])

    assert count() == 2
    dec.iou_threshold = 1.0
    assert count() == 2          # stale: the traced function still uses 0.5
    model.predict_function = None
    assert count() == 3          # retraced: NMS off


def test_adapter_retraces_only_when_value_changes_and_restores():
    adapter = KerasRetinaNet()
    adapter.ensure_loaded()
    model = adapter._model
    img = ImageInput(image=Image.new("RGB", (64, 64)))
    adapter.predict(img, 0.5)                       # default 0.5: traced once
    traced = model.predict_function
    adapter.predict(img, 0.5, Options(nms_iou=0.5))
    assert model.predict_function is traced        # unchanged value: no retrace
    adapter.predict(img, 0.5, Options(nms_iou=1.0))
    assert model.prediction_decoder.iou_threshold == 0.5   # restored
    assert adapter.nms_knob_works is True
