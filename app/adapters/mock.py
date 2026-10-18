"""Mock adapter: deterministic fake detections derived from the image size. Needs no ML library."""
from __future__ import annotations

import time

from ..postprocess import nms
from ..schemas import ImageInput, Options, RawOutput
from .base import Adapter

# (fx1, fy1, fx2, fy2, score, coco id) as fractions of the image size.
_PERSON_CAR = [
    (0.10, 0.30, 0.22, 0.85, 0.94, 1),
    (0.55, 0.35, 0.65, 0.88, 0.81, 1),
    (0.30, 0.55, 0.52, 0.80, 0.90, 3),
    (0.60, 0.62, 0.92, 0.90, 0.72, 3),
    (0.75, 0.10, 0.80, 0.22, 0.28, 1),   # small, low score: the threshold slider has something to do
    (0.31, 0.56, 0.53, 0.81, 0.40, 3),   # near-duplicate of the first car: NMS has something to do
    (0.02, 0.05, 0.05, 0.10, 0.12, 3),   # tiny, very low score
]
_PEOPLE_ONLY = [(0.10, 0.30, 0.22, 0.85, 0.91, 1), (0.55, 0.35, 0.65, 0.88, 0.58, 1),
                (0.11, 0.31, 0.23, 0.86, 0.35, 1), (0.77, 0.12, 0.81, 0.24, 0.10, 1)]


class MockAdapter(Adapter):
    def __init__(self, algorithm: str, framework: str) -> None:
        super().__init__()
        self.algorithm, self.framework = algorithm, framework
        self.model_name = f"mock-{algorithm}-{framework}"
        self.mock = True
        if algorithm == "hog_svm":
            self.score_meaning = "rescaled SVM score"
            self.classes_supported = 1
        else:
            self.parameters = {"faster_rcnn": 41_755_286, "retinanet": 34_014_999}[algorithm]
            self.model_size_mb = round(self.parameters * 4 / 1e6, 1)

    def _load(self) -> None:
        pass

    def _predict(self, image_input: ImageInput, options: Options) -> RawOutput:
        time.sleep(0.01)
        w, h = image_input.width, image_input.height
        rows = _PEOPLE_ONLY if self.algorithm == "hog_svm" else _PERSON_CAR
        if self.algorithm == "faster_rcnn":
            rows = rows[:5] + rows[6:]  # slightly different set from RetinaNet
        boxes = [[fx1 * w, fy1 * h, fx2 * w, fy2 * h] for fx1, fy1, fx2, fy2, _, _ in rows]
        scores = [r[4] for r in rows]
        labels = [r[5] for r in rows]
        if options.nms_iou < 1.0:
            keep = nms(boxes, scores, options.nms_iou, labels)
            boxes, scores, labels = [boxes[i] for i in keep], [scores[i] for i in keep], [labels[i] for i in keep]
        settings = {"nms_iou": options.nms_iou, "nms_applied": True}
        if self.algorithm == "hog_svm":
            settings.update(hog_stride=options.hog_stride, hog_scale=options.hog_scale,
                            hit_threshold=options.hit_threshold)
        return RawOutput(boxes, scores, labels, settings)

