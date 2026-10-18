"""Whitelisted code excerpts: slices between `# excerpt: <id>` and `# endexcerpt` markers."""
from __future__ import annotations

import re

from .config import ROOT

WHITELIST = {
    "hog_predict": "app/adapters/hog_opencv.py",
    "fasterrcnn_pytorch": "app/adapters/torchvision_detectors.py",
    "retinanet_pytorch": "app/adapters/torchvision_detectors.py",
    "nms_reference": "app/postprocess.py",
    "postprocess_build": "app/postprocess.py",
    "backend_contract": "app/adapters/base.py",
    "retinanet_tensorflow": "app/adapters/keras_retinanet.py",
    "train_pytorch": "scripts/train_cnn_pytorch.py",
    "train_keras": "scripts/train_cnn_keras.py",
}

_START = re.compile(r"^\s*#\s*excerpt:\s*([A-Za-z0-9_]+)")
_END = re.compile(r"^\s*#\s*endexcerpt\b")


def extract(excerpt_id: str) -> dict | None:
    """Return {id, file, start_line, language, code} or None for ids outside the whitelist."""
    rel = WHITELIST.get(excerpt_id)
    if rel is None:
        return None
    path = ROOT / rel
    if not path.is_file():
        return None
    regions: list[tuple[int, list[str]]] = []
    current: list[str] | None = None
    start = 0
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        m = _START.match(line)
        if m and m.group(1) == excerpt_id:
            current, start = [], number + 1
            continue
        if current is not None and _END.match(line):
            regions.append((start, current))
            current = None
            continue
        if current is not None:
            current.append(line)
    if not regions:
        return None
    # several regions of one id are joined by a line "# ..."
    parts: list[str] = []
    for i, (_, lines) in enumerate(regions):
        if i:
            parts.append("# ...")
        parts.extend(lines)
    return {"id": excerpt_id, "file": rel, "start_line": regions[0][0], "language": "python",
            "code": "\n".join(parts).rstrip("\n")}

