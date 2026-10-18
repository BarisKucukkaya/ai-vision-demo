"""Real-weights validation: POST /compare at threshold 0.5 and check the results make sense."""
import argparse
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.postprocess import iou  # noqa: E402


def best_box(result, label):
    boxes = [d for d in result["detections"] if d["label"] == label]
    return max(boxes, key=lambda d: d["confidence"]) if boxes else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--expect", default="person,car", help="labels every deep detector must find")
    parser.add_argument("--combos", help="comma list; default all registered")
    parser.add_argument("--allow-mock", action="store_true")
    args = parser.parse_args()
    expected = [e.strip() for e in args.expect.split(",") if e.strip()]

    path = Path(args.image)
    if not path.is_file():
        sys.exit(f"error: image not found: {path}")
    params = {"threshold": 0.5}
    if args.combos:
        params["combos"] = args.combos
    try:
        resp = httpx.post(f"{args.url}/compare", params=params, timeout=600,
                          files={"file": (path.name, path.read_bytes(), "application/octet-stream")})
    except httpx.HTTPError as exc:
        sys.exit(f"error: cannot reach {args.url}: {exc}")
    if resp.status_code != 200:
        sys.exit(f"error: /compare returned {resp.status_code}: {resp.text}")
    data = resp.json()
    w, h = data["image"]["width"], data["image"]["height"]
    failures: list[str] = []
    print(f"{'combo':24} {'ok':3} {'ms':>9} {'boxes':>5}  found")
    ok_results = {}
    for combo, r in data["results"].items():
        if not r.get("ok"):
            print(f"{combo:24} --  {'':>9} {'':>5}  {r.get('error')}")
            if not r.get("unavailable"):
                failures.append(f"{combo}: {r.get('error')}")
            continue
        ok_results[combo] = r
        labels = sorted({d["label"] for d in r["detections"]})
        print(f"{combo:24} yes {r['inference_ms']:9.1f} {len(r['detections']):5}  {', '.join(labels)}")
        if r["mock"] and not args.allow_mock:
            failures.append(f"{combo}: mock/test-model result (use --allow-mock to accept)")
        for d in r["detections"]:
            x1, y1, x2, y2 = d["box"]
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                failures.append(f"{combo}: box outside image {d['box']}")
                break
        needed = ["person"] if combo.startswith("hog_svm") else expected
        for label in needed:
            if best_box(r, label) is None and not (r["mock"] and args.allow_mock):
                failures.append(f"{combo}: expected label '{label}' not found at threshold 0.5")
    a, b = ok_results.get("retinanet/pytorch"), ok_results.get("retinanet/tensorflow")
    if a and b:
        # With many same-class objects the two frameworks may rank different instances first, so each
        # side's best box of a label must overlap SOME box of that label on the other side (IoU >= 0.5).
        def best_overlap(box, other, label):
            return max((iou(box, d["box"]) for d in other["detections"] if d["label"] == label), default=0.0)

        for label in expected:
            ba, bb = best_box(a, label), best_box(b, label)
            if ba and bb:
                value = min(best_overlap(ba["box"], b, label), best_overlap(bb["box"], a, label))
                print(f"retinanet pytorch vs tensorflow, best '{label}' vs any '{label}' on the other side: IoU {value:.2f}")
                if value < 0.5:
                    failures.append(f"RetinaNet frameworks disagree on '{label}' (IoU {value:.2f} < 0.5)")
    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("\nsmoke test passed")


if __name__ == "__main__":
    main()
