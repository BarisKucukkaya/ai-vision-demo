import io
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def png_bytes(w=320, h=240, color=(120, 120, 120), fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, fmt)
    return buf.getvalue()


def make_client(monkeypatch, **env):
    from fastapi.testclient import TestClient

    base = {"DEMO_MOCK": "1", "DEMO_PRELOAD": "0", "DEMO_COMBOS": "", "DEMO_REMOTES": ""}
    base.update(env)
    for key, value in base.items():
        monkeypatch.setenv(key, value)
    from app.main import app

    return TestClient(app)


@pytest.fixture
def client(monkeypatch):
    with make_client(monkeypatch) as c:
        yield c
