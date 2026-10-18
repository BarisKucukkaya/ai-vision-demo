"""Record /compare results (floor threshold 0.05) for every photo in samples/ as offline fallback data."""
import argparse
import datetime
import json
import platform
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "app" / "static" / "demo" / "data" / "recorded"
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--samples", type=Path, default=ROOT / "samples")
    args = parser.parse_args()

    images = sorted(p for p in args.samples.glob("*") if p.suffix.lower() in SUFFIXES)
    if not images:
        sys.exit(f"error: no photos in {args.samples} (see samples/README.txt)")
    args.out.mkdir(parents=True, exist_ok=True)
    machine = f"{platform.platform()}, Python {platform.python_version()}"
    for path in images:
        try:
            resp = httpx.post(f"{args.url}/compare", params={"threshold": 0.05}, timeout=900,
                              files={"file": (path.name, path.read_bytes(), "application/octet-stream")})
        except httpx.HTTPError as exc:
            sys.exit(f"error: cannot reach {args.url}: {exc}")
        if resp.status_code != 200:
            sys.exit(f"error: /compare returned {resp.status_code}: {resp.text}")
        data = resp.json()
        results = {}
        for combo, r in data["results"].items():
            if r.get("ok"):
                results[combo] = {
                    "model": r["model"], "inference_ms": r["inference_ms"], "device": r["device"],
                    "mock": r["mock"], "score_meaning": r["score_meaning"], "settings": r["settings"],
                    "model_info": r["model_info"], "detections": r["detections"],
                }
        record = {
            "image": {"name": path.name, **data["image"]},
            "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "machine": machine,
            "results": results,
        }
        target = args.out / f"{path.stem}.json"
        target.write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(f"wrote {target} ({', '.join(results)})")


if __name__ == "__main__":
    main()
