"""Write app/static/demo/data/code_excerpts.json (offline fallback of GET /api/code/{id})."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import codeexcerpts  # noqa: E402

DEFAULT_OUT = ROOT / "app" / "static" / "demo" / "data" / "code_excerpts.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    snapshot = {i: codeexcerpts.extract(i) for i in codeexcerpts.WHITELIST}
    missing = [i for i, v in snapshot.items() if v is None]
    if missing:
        sys.exit(f"error: no excerpt markers found for: {', '.join(missing)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snapshot, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out} ({len(snapshot)} excerpts)")
    # Line counts behind the panel's shared vs framework-specific bar (generated, never typed).
    groups = {
        "shared": ["app/adapters/base.py", "app/postprocess.py", "app/schemas.py", "app/registry.py", "app/main.py"],
        "framework": ["app/adapters/hog_opencv.py", "app/adapters/torchvision_detectors.py", "app/adapters/keras_retinanet.py"],
    }
    files = {
        group: [{"file": f, "lines": sum(1 for line in (ROOT / f).read_text(encoding="utf-8").splitlines() if line.strip())} for f in names]
        for group, names in groups.items()
    }
    target = args.out.parent / "code_files.json"
    target.write_text(json.dumps(files, indent=1), encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
