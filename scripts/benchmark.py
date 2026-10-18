"""Sequential /predict calls at threshold 0.5; prints server-side inference_ms statistics per combo."""
import argparse
import json
import platform
import statistics
import sys
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--combos", help="comma list; default all registered")
    parser.add_argument("--json", type=Path, help="write the results to this file")
    args = parser.parse_args()

    path = Path(args.image)
    if not path.is_file():
        sys.exit(f"error: image not found: {path}")
    raw = path.read_bytes()
    try:
        health = httpx.get(f"{args.url}/health", timeout=10).json()
    except httpx.HTTPError as exc:
        sys.exit(f"error: cannot reach {args.url}: {exc}")
    combos = [c.strip() for c in args.combos.split(",")] if args.combos else list(health["backends"])

    results, device = {}, {}
    for combo in combos:
        algorithm, framework = combo.split("/")
        params = {"algorithm": algorithm, "framework": framework, "threshold": 0.5}
        times, last = [], None
        for i in range(args.warmup + args.runs):
            resp = httpx.post(f"{args.url}/predict", params=params, timeout=600,
                              files={"file": (path.name, raw, "application/octet-stream")})
            if resp.status_code != 200:
                print(f"{combo}: skipped ({resp.status_code}: {resp.text[:120]})")
                times = []
                break
            last = resp.json()
            if i >= args.warmup:
                times.append(last["inference_ms"])
        if not times:
            continue
        times.sort()
        p95 = times[min(len(times) - 1, int(round(0.95 * (len(times) - 1))))]
        results[combo] = {
            "median": round(statistics.median(times), 1), "mean": round(statistics.mean(times), 1),
            "p95": round(p95, 1), "min": round(times[0], 1), "max": round(times[-1], 1),
            "parameters": last["model_info"]["parameters"], "model_size_mb": last["model_info"]["model_size_mb"],
            "device": last["device"],
        }
        device[combo] = last["device"]

    machine = {"os": platform.platform(), "python": platform.python_version(), "processor": platform.processor(),
               "device": device}
    print(f"machine: {machine['os']}, Python {machine['python']}, {machine['processor']}")
    print(f"{'combo':24} {'median':>9} {'mean':>9} {'p95':>9} {'min':>9} {'max':>9}  device")
    for combo, r in results.items():
        print(f"{combo:24} {r['median']:9.1f} {r['mean']:9.1f} {r['p95']:9.1f} {r['min']:9.1f} {r['max']:9.1f}  {r['device']}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"machine": machine, "image": path.name, "runs": args.runs, "results": results}, indent=1),
                             encoding="utf-8")
        print(f"wrote {args.json}")
    if not results:
        sys.exit(1)


if __name__ == "__main__":
    main()
