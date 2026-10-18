import json
import re
import subprocess
import sys

import pytest

from app import codeexcerpts
from tests.conftest import ROOT


def read_slices(rel, excerpt_id):
    lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()
    out, current, first = [], None, None
    for number, line in enumerate(lines, 1):
        m = re.match(r"^\s*#\s*excerpt:\s*(\w+)", line)
        if m and m.group(1) == excerpt_id:
            current = []
            first = first or number + 1
        elif current is not None and re.match(r"^\s*#\s*endexcerpt", line):
            out.append(current)
            current = None
        elif current is not None:
            current.append(line)
    return first, out


@pytest.mark.parametrize("excerpt_id", sorted(codeexcerpts.WHITELIST))
def test_whitelisted_ids_return_exact_slice(client, excerpt_id):
    j = client.get(f"/api/code/{excerpt_id}").json()
    first, regions = read_slices(codeexcerpts.WHITELIST[excerpt_id], excerpt_id)
    expected = "\n# ...\n".join("\n".join(r) for r in regions).rstrip("\n")
    assert j["code"] == expected and j["start_line"] == first
    assert j["file"] == codeexcerpts.WHITELIST[excerpt_id] and j["language"] == "python"
    assert "excerpt" not in j["code"].replace("excerpt_", "")  # no marker lines in the slice


def test_start_line_is_real_file_line(client):
    j = client.get("/api/code/nms_reference").json()
    lines = (ROOT / j["file"]).read_text(encoding="utf-8").splitlines()
    assert lines[j["start_line"] - 1] == j["code"].splitlines()[0]


@pytest.mark.parametrize("bad", ["nope", "..%2Fapp%2Fmain.py", "app/main.py", "%2e%2e%2f%2e%2e%2fREADME.md", "main"])
def test_unknown_id_is_404(client, bad):
    assert client.get(f"/api/code/{bad}").status_code == 404


def test_every_excerpt_has_content():
    for excerpt_id in codeexcerpts.WHITELIST:
        assert codeexcerpts.extract(excerpt_id)["code"].strip(), excerpt_id


def test_snapshot_equals_endpoint(tmp_path):
    out = tmp_path / "snap.json"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "snapshot_code.py"), "--out", str(out)], check=True, cwd=ROOT)
    snap = json.loads(out.read_text(encoding="utf-8"))
    for excerpt_id in codeexcerpts.WHITELIST:
        assert snap[excerpt_id] == codeexcerpts.extract(excerpt_id)
