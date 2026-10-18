"""hog_svm/opencv: classical HOG features + the default linear SVM people detector."""
from __future__ import annotations

import numpy as np

from ..postprocess import nms
from ..schemas import ImageInput, Options, RawOutput
from .base import Adapter, BackendUnavailable

MAX_SIDE = 1000  # downscale larger images for speed, boxes are mapped back
PIN_HINT = "Install OpenCV with: pip install \"opencv-python-headless>=4.8,<5\" (OpenCV 5 has no HOGDescriptor)."


class HogOpenCV(Adapter):
    algorithm = "hog_svm"
    framework = "opencv"
    model_name = "HOG + linear SVM (OpenCV default people detector)"
    score_meaning = "rescaled SVM score"
    classes_supported = 1

    def __init__(self) -> None:
        super().__init__()
        self._cv2 = None
        self._hog = None

    def _load(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise BackendUnavailable(f"OpenCV is not installed. {PIN_HINT}") from exc
        if not hasattr(cv2, "HOGDescriptor"):
            raise BackendUnavailable(f"This OpenCV build ({cv2.__version__}) has no HOGDescriptor. {PIN_HINT}")
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._cv2, self._hog = cv2, hog
        self.device = "cpu"

    # excerpt: hog_predict
    def _predict(self, image_input: ImageInput, options: Options) -> RawOutput:
        cv2 = self._cv2
        rgb = np.asarray(image_input.image)                    # PIL gives RGB
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)             # OpenCV expects BGR
        h, w = bgr.shape[:2]
        shrink = min(1.0, MAX_SIDE / max(w, h))
        if shrink < 1.0:
            bgr = cv2.resize(bgr, (round(w * shrink), round(h * shrink)))
        s = options.hog_stride
        rects, weights = self._hog.detectMultiScale(
            bgr, winStride=(s, s), padding=(8, 8), scale=options.hog_scale, hitThreshold=options.hit_threshold,
        )
        scores = np.asarray(weights, dtype=float).reshape(-1)  # empty tuple when nothing is found
        rects = np.asarray(rects, dtype=float).reshape(-1, 4) / shrink
        boxes = np.column_stack([rects[:, 0], rects[:, 1], rects[:, 0] + rects[:, 2], rects[:, 1] + rects[:, 3]])
        scores = np.clip(scores, 0.0, 3.0) / 3.0               # SVM value -> 0..1 display scale
        labels = [1] * len(scores)                             # the default detector only finds people
        if options.nms_iou < 1.0 and len(boxes):
            keep = nms(boxes.tolist(), scores.tolist(), options.nms_iou)
            boxes, scores, labels = boxes[keep], scores[keep], [labels[i] for i in keep]
    # endexcerpt
        settings = {
            "nms_iou": options.nms_iou, "nms_applied": options.nms_iou < 1.0,
            "hog_stride": s, "hog_scale": options.hog_scale, "hit_threshold": options.hit_threshold,
        }
        return RawOutput(boxes, scores, labels, settings)
