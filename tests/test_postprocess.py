from app.postprocess import build_detections, iou, nms

NAMES = {1: "person", 3: "car"}


def build(boxes, scores, labels, thr=0.5, w=100, h=100, **kw):
    return build_detections(boxes, scores, labels, NAMES, thr, w, h, **kw)


def test_threshold_and_sorting():
    dets = build([[0, 0, 10, 10], [20, 20, 40, 40], [50, 50, 60, 60]], [0.6, 0.9, 0.4], [1, 3, 1])
    assert [d.label for d in dets] == ["car", "person"]
    assert [d.confidence for d in dets] == [0.9, 0.6]


def test_threshold_is_inclusive():
    assert len(build([[0, 0, 10, 10]], [0.5], [1])) == 1


def test_clipping_to_image():
    d = build([[-5, -5, 150, 50]], [0.9], [1])[0]
    assert d.box == [0.0, 0.0, 100.0, 50.0]


def test_sub_pixel_boxes_dropped():
    assert build([[10, 10, 10.5, 40], [10, 10, 40, 10.9]], [0.9, 0.9], [1, 1]) == []
    assert build([[150, 10, 200, 40]], [0.9], [1]) == []  # fully outside: zero width after clipping


def test_unknown_label_fallback():
    assert build([[0, 0, 10, 10]], [0.9], [77])[0].label == "class_77"


def test_cap_of_100_and_rounding():
    boxes = [[0, 0, 10.123, 10.987]] * 150
    dets = build(boxes, [0.123456] * 150, [1] * 150, thr=0.1)
    assert len(dets) == 100
    assert dets[0].confidence == 0.1235 and dets[0].box == [0.0, 0.0, 10.1, 11.0]


def test_iou_cases():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - 50 / 150) < 1e-9
    assert iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0


def test_nms_disjoint_and_identical():
    assert nms([[0, 0, 5, 5], [10, 10, 15, 15]], [0.5, 0.9], 0.5) == [1, 0]
    assert nms([[0, 0, 10, 10], [0, 0, 10, 10]], [0.5, 0.9], 0.5) == [1]


def test_nms_partial_overlap_and_boundary():
    boxes = [[0, 0, 10, 10], [5, 0, 15, 10]]  # IoU = 1/3
    scores = [0.9, 0.8]
    assert nms(boxes, scores, 0.5) == [0, 1]
    assert nms(boxes, scores, 0.3) == [0]
    assert nms(boxes, scores, 50 / 150) == [0, 1]  # suppress only when IoU is greater than the threshold


def test_nms_one_means_off():
    assert nms([[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.8], 1.0) == [0, 1]


def test_nms_per_label():
    boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
    assert nms(boxes, [0.9, 0.8], 0.5, labels=[1, 3]) == [0, 1]
    assert nms(boxes, [0.9, 0.8], 0.5, labels=[1, 1]) == [0]


def test_iou_matches_shared_fixture():
    import json
    from pathlib import Path

    cases = json.loads((Path(__file__).parent / "fixtures" / "iou_cases.json").read_text())
    for c in cases:
        assert abs(iou(c["a"], c["b"]) - c["iou"]) < 1e-9
