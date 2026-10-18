"""Builds one adapter per combo from DEMO_COMBOS / DEMO_REMOTES / DEMO_MOCK."""
from __future__ import annotations

from . import config
from .adapters.base import Adapter
from .adapters.mock import MockAdapter
from .adapters.remote import RemoteAdapter

ALGORITHMS = {"hog_svm": ["opencv"], "faster_rcnn": ["pytorch"], "retinanet": ["pytorch", "tensorflow"]}
UNAVAILABLE = {"faster_rcnn/tensorflow": "KerasHub provides no Faster R-CNN"}
KNOWN_FRAMEWORKS = {"opencv", "pytorch", "tensorflow"}
ORDER = ["hog_svm", "faster_rcnn", "retinanet"]


def split_combo(combo: str) -> tuple[str, str]:
    algorithm, _, framework = combo.partition("/")
    return algorithm, framework


def valid_combos() -> list[str]:
    return [f"{a}/{f}" for a in ORDER for f in ALGORITHMS[a]]


def _local(combo: str) -> Adapter:
    algorithm, framework = split_combo(combo)
    if config.mock_mode():
        return MockAdapter(algorithm, framework)
    if combo == "hog_svm/opencv":
        from .adapters.hog_opencv import HogOpenCV
        return HogOpenCV()
    if combo == "faster_rcnn/pytorch":
        from .adapters.torchvision_detectors import FasterRCNNPyTorch
        return FasterRCNNPyTorch()
    if combo == "retinanet/pytorch":
        from .adapters.torchvision_detectors import RetinaNetPyTorch
        return RetinaNetPyTorch()
    if combo == "retinanet/tensorflow":
        from .adapters.keras_retinanet import KerasRetinaNet
        return KerasRetinaNet()
    raise ValueError(f"unknown combo {combo}")


def build_adapters() -> dict[str, Adapter]:
    remotes = config.demo_remotes()
    wanted = [c for c in valid_combos() if c in config.demo_combos() or c in remotes]
    adapters: dict[str, Adapter] = {}
    for combo in wanted:
        if combo in remotes:
            algorithm, framework = split_combo(combo)
            adapters[combo] = RemoteAdapter(algorithm, framework, remotes[combo])
        else:
            adapters[combo] = _local(combo)
    return adapters
