"""Framework-independent post-processing shared by every algorithm."""
from __future__ import annotations

from typing import Sequence

from .schemas import Detection


# excerpt: postprocess_build
def build_detections(boxes, scores, labels, names, threshold, width, height, max_detections=100):
    """Threshold, clip, name and sort raw model output (boxes are xyxy in original pixels)."""
    out: list[Detection] = []
    for box, score, label in zip(boxes, scores, labels):
        score = float(score)
        if score < threshold:
            continue
        x1 = min(max(float(box[0]), 0.0), width)
        y1 = min(max(float(box[1]), 0.0), height)
        x2 = min(max(float(box[2]), 0.0), width)
        y2 = min(max(float(box[3]), 0.0), height)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue  # thinner than one pixel after clipping
        name = names.get(int(label), f"class_{int(label)}")
        out.append(Detection(name, round(score, 4), [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]))
    out.sort(key=lambda d: d.confidence, reverse=True)
    return out[:max_detections]
# endexcerpt


# excerpt: nms_reference
def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two [x1, y1, x2, y2] boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, iou_threshold, labels=None):
    """Greedy NMS: keep the best box, drop boxes with IoU above the threshold, repeat."""
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    for i in order:
        if all(
            iou(boxes[i], boxes[j]) <= iou_threshold
            for j in keep
            if labels is None or labels[i] == labels[j]
        ):
            keep.append(i)
    return keep
# endexcerpt

