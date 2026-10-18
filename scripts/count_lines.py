"""Count non-blank, non-comment lines in the train_pytorch / train_keras excerpts."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import codeexcerpts  # noqa: E402

DEFAULT_OUT = ROOT / "app" / "static" / "demo" / "data" / "code_lines.json"


def count(excerpt_id: str) -> int:
    code = codeexcerpts.extract(excerpt_id)["code"]
    return sum(1 for line in code.splitlines() if line.strip() and not line.strip().startswith("#"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write the numbers to this file")
    args = parser.parse_args()
    result = {"train_pytorch": count("train_pytorch"), "train_keras": count("train_keras")}
    print(result)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
