"""Start the gateway (8000) and two workers (8101 torch, 8102 tensorflow), each in its own virtual environment."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-python", required=True, help="interpreter of venv-gateway")
    parser.add_argument("--torch-python", required=True, help="interpreter of venv-torch")
    parser.add_argument("--tf-python", required=True, help="interpreter of venv-tf")
    parser.add_argument("--mock", action="store_true", help="DEMO_MOCK=1 everywhere (no weights needed)")
    args = parser.parse_args()
    args.gateway_python, args.torch_python, args.tf_python = (str(Path(p).resolve()) for p in (args.gateway_python, args.torch_python, args.tf_python))
    for p in (args.gateway_python, args.torch_python, args.tf_python):
        if not Path(p).is_file():
            sys.exit(f"error: interpreter not found: {p}")

    base = dict(os.environ, DEMO_MOCK="1" if args.mock else "0")
    torch_url, tf_url = "http://127.0.0.1:8101", "http://127.0.0.1:8102"
    plan = [
        ("worker-torch", args.torch_python, 8101,
         {"DEMO_COMBOS": "faster_rcnn/pytorch,retinanet/pytorch", "DEMO_REMOTES": ""}),
        ("worker-tf", args.tf_python, 8102, {"DEMO_COMBOS": "retinanet/tensorflow", "DEMO_REMOTES": ""}),
        ("gateway", args.gateway_python, 8000,
         {"DEMO_COMBOS": "hog_svm/opencv",
          "DEMO_REMOTES": f"faster_rcnn/pytorch={torch_url},retinanet/pytorch={torch_url},retinanet/tensorflow={tf_url}"}),
    ]
    procs = []
    try:
        for name, python, port, env in plan:
            proc = subprocess.Popen([python, "-m", "uvicorn", "app.main:app", "--port", str(port)],
                                    cwd=ROOT, env=dict(base, **env))
            procs.append(proc)
            for _ in range(300):  # /health answers while models are still loading
                if proc.poll() is not None:
                    sys.exit(f"error: {name} exited with code {proc.returncode}")
                try:
                    if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(0.5)
            else:
                sys.exit(f"error: {name} did not answer /health")
            print(f"{name} ready on port {port}")
        print("open http://127.0.0.1:8000  (Ctrl+C stops everything)")
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
