import pytest

from app.coco_classes import COCO_CATEGORIES


def test_table_shape():
    assert len(COCO_CATEGORIES) == 80
    assert COCO_CATEGORIES[1] == "person" and COCO_CATEGORIES[3] == "car" and COCO_CATEGORIES[10] == "traffic light"
    assert all("_" not in name for name in COCO_CATEGORIES.values())


def test_matches_torchvision():
    pytest.importorskip("torchvision")
    from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights

    cats = RetinaNet_ResNet50_FPN_Weights.COCO_V1.meta["categories"]
    assert COCO_CATEGORIES == {i: n for i, n in enumerate(cats) if n not in ("__background__", "N/A")}


def test_matches_keras_hub():
    pytest.importorskip("keras_hub")
    from keras_hub.src.utils.coco.coco_utils import COCO_NAMES  # private path: tests only

    names = {k: v.replace("_", " ") for k, v in COCO_NAMES.items()}
    for i, name in COCO_CATEGORIES.items():
        assert names[i] == name
