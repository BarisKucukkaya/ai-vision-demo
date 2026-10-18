"""retinanet/tensorflow: KerasHub RetinaNet. Everything TensorFlow/Keras is imported inside _load()."""
from __future__ import annotations

import os
import time

import numpy as np
from PIL import Image

from .. import config
from ..schemas import ImageInput, Options, RawOutput
from .base import Adapter, BackendUnavailable

FLOOR = 0.05
DEFAULT_SIZE = (800, 800)  # (height, width) of the model input when it cannot be read from the preprocessor


# excerpt: retinanet_tensorflow
def letterbox(image: Image.Image, target_w: int, target_h: int):
    """Resize keeping the aspect ratio and centre on a black canvas. Returns (canvas, scale, ox, oy)."""
    w, h = image.size
    scale = min(target_w / w, target_h / h)
    nw, nh = round(w * scale), round(h * scale)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    ox, oy = int((target_w - nw) / 2), int((target_h - nh) / 2)
    canvas[oy:oy + nh, ox:ox + nw] = np.asarray(image.resize((nw, nh), Image.BILINEAR))
    return canvas, scale, ox, oy


def to_original(boxes_xyxy, scale, ox, oy):
    """Inverse of letterbox for boxes given as xyxy in model-input pixels."""
    boxes = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4).copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - ox) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - oy) / scale
    return boxes
# endexcerpt


def build_random_model():
    """A tiny RetinaNet with random weights (test only, no download). VERIFY against the installed keras-hub."""
    import keras_hub

    encoder = keras_hub.models.ResNetBackbone(
        input_conv_filters=[64], input_conv_kernel_sizes=[7],
        stackwise_num_filters=[64, 128, 256, 512], stackwise_num_blocks=[1, 1, 1, 1],
        stackwise_num_strides=[1, 2, 2, 2], block_type="basic_block", image_shape=(None, None, 3),
    )
    backbone = keras_hub.models.RetinaNetBackbone(image_encoder=encoder, min_level=3, max_level=7, use_p5=False)
    converter = keras_hub.layers.RetinaNetImageConverter(image_size=(256, 256), scale=1 / 255, offset=None)
    preprocessor = keras_hub.models.RetinaNetObjectDetectorPreprocessor(image_converter=converter)
    return keras_hub.models.RetinaNetObjectDetector(backbone=backbone, num_classes=91, preprocessor=preprocessor)


class KerasRetinaNet(Adapter):
    algorithm = "retinanet"
    framework = "tensorflow"
    model_name = "RetinaNet ResNet50-FPN COCO (KerasHub: retinanet_resnet50_fpn_coco)"

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._keras = None
        self.bounding_box_format = "yxyx"
        self.input_hw = DEFAULT_SIZE
        self.nms_knob_works = True

    def _load(self) -> None:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        try:
            import keras
            import keras_hub
            import tensorflow as tf
        except ImportError as exc:
            raise BackendUnavailable(
                "TensorFlow/KerasHub are not installed. Install with: pip install -r requirements-tensorflow.txt"
            ) from exc
        random = config.random_weights()
        preset = config.tf_preset()
        model = build_random_model() if random else keras_hub.models.ImageObjectDetector.from_preset(preset)
        self._keras, self._model = keras, model
        self.bounding_box_format = getattr(model, "bounding_box_format", "yxyx")
        try:
            h, w = model.preprocessor.image_converter.image_size
            self.input_hw = (int(h), int(w))
        except Exception:
            self.input_hw = DEFAULT_SIZE
        # excerpt: retinanet_tensorflow
        model.prediction_decoder.confidence_threshold = FLOOR   # the UI slider filters afterwards
        # endexcerpt
        self.parameters = int(model.count_params())
        self.model_size_mb = round(self.parameters * 4 / 1e6, 1)  # float32
        gpus = tf.config.list_physical_devices("GPU")
        self.device = "cuda" if gpus else "cpu"
        if random:
            self.mock = True
            self.model_name += " - RANDOM weights - test only"
        grey = Image.new("RGB", (self.input_hw[1], self.input_hw[0]), (128, 128, 128))
        self._run(letterbox(grey, self.input_hw[1], self.input_hw[0])[0])  # warm-up

    # excerpt: retinanet_tensorflow
    def _run(self, canvas):
        """Model call on a letterboxed HWC 0-255 canvas. Returns xyxy boxes (model pixels), scores, labels."""
        th, tw = canvas.shape[:2]
        out = self._model.predict(canvas[None].astype("float32"), verbose=0)
        n = int(out["num_detections"][0])
        boxes = np.asarray(out["boxes"][0][:n])
        boxes = self._keras.utils.bounding_boxes.convert_format(
            boxes, source=self.bounding_box_format, target="xyxy", height=th, width=tw,
        )
        return np.asarray(boxes).reshape(-1, 4), np.asarray(out["confidence"][0][:n]), np.asarray(out["labels"][0][:n])
    # endexcerpt

    def _set_decoder_iou(self, value: float) -> None:
        """Change the decoder's NMS IoU. Keras traces predict() once and bakes the value in (verified with a
        crafted-box test), so the compiled predict function is dropped whenever the value really changes."""
        decoder = self._model.prediction_decoder
        if decoder.iou_threshold != value:
            decoder.iou_threshold = value
            self._model.predict_function = None  # Keras 3 rebuilds it on the next predict()

    def _predict(self, image_input: ImageInput, options: Options) -> RawOutput:
        th, tw = self.input_hw
        canvas, scale, ox, oy = letterbox(image_input.image, tw, th)
        previous = self._model.prediction_decoder.iou_threshold
        applied = self.nms_knob_works
        changed = applied and previous != options.nms_iou
        if applied:
            self._set_decoder_iou(options.nms_iou)
        try:
            if changed:
                self._run(canvas)  # retrace with the new value, outside the timed call
            start = time.perf_counter()
            boxes, scores, labels = self._run(canvas)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        finally:
            if applied:
                self._set_decoder_iou(previous)
                if changed:
                    self._run(canvas)  # retrace back so the next default request is not slowed down
        boxes = to_original(boxes, scale, ox, oy)
        return RawOutput(boxes, scores, labels, {"nms_iou": options.nms_iou, "nms_applied": applied}, inference_ms=elapsed_ms)
