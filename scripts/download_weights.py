"""Load each model once through its adapter so the weight caches fill. HOG needs no download."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TARGETS = {
    "retinanet-pytorch": "retinanet/pytorch",
    "faster-rcnn-pytorch": "faster_rcnn/pytorch",
    "retinanet-tensorflow": "retinanet/tensorflow",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma list of: " + ", ".join(TARGETS))
    args = parser.parse_args()
    names = [n.strip() for n in args.only.split(",")] if args.only else list(TARGETS)
    unknown = [n for n in names if n not in TARGETS]
    if unknown:
        sys.exit(f"error: unknown target(s) {unknown}; choose from {list(TARGETS)}")

    from app import config, registry

    if config.mock_mode():
        sys.exit("error: DEMO_MOCK=1 is set; unset it to download real weights.")
    failed = []
    for name in names:
        combo = TARGETS[name]
        adapter = registry._local(combo)
        try:
            adapter.ensure_loaded()
            info = adapter.info()
            print(f"OK   {combo}: {info['parameters']:,} parameters, {info['model_size_mb']} MB (float32), device {info['device']}")
        except Exception as exc:
            failed.append(combo)
            print(f"FAIL {combo}: {exc}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
