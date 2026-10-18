# AI algorithms compared

Running code example for SE 4458, Assignment 1 (topic: *Image Processing AI Algorithms ex. TensorFlow, PyTorch*).
One small FastAPI service serves a web page that runs **three image-AI algorithms on the same photo** and compares them live:

| Algorithm | Family | Runs on |
| --- | --- | --- |
| HOG + SVM | classic machine learning | OpenCV (same whichever framework is selected) |
| Faster R-CNN | deep, two-stage | PyTorch / torchvision (KerasHub has no Faster R-CNN) |
| RetinaNet | deep, one-stage | PyTorch / torchvision and TensorFlow / KerasHub |

The page shows boxes, a threshold sweep, class coverage, size buckets, an agreement matrix, the NMS effect and cost,
next to the **real code** (never retyped) that makes the difference. It has no accuracy numbers on purpose: there is no ground truth.

## Requirements

* **Python 3.11 or 3.12** (developed and tested with 3.12.10 on Windows 11, CPU only). Check with `python --version` (`python3 --version` on Linux/macOS).
  Newer versions (3.13+) may not have TensorFlow wheels yet: use 3.11 or 3.12.
* About **5 GB of free disk** for PyTorch + TensorFlow + the downloaded weights, and internet **once** to install packages and download the weights.
  After that the page works fully offline (no CDN, fonts or images from outside).
* A browser. **Node.js 20+** is only needed for the optional JavaScript tests.
* Everything below is run from the project folder (the one containing this README). You do not need to "activate" a virtual environment:
  every command calls the interpreter inside the environment directly.

Not tested: Linux, macOS, Apple Silicon and GPUs (the bash commands below are the same steps with Unix paths). `DEMO_DEVICE` accepts `auto`, `cpu` or `cuda`.

**OpenCV must stay below version 5**: `opencv-python-headless` 5.0 has no `cv2.HOGDescriptor`, so HOG + SVM would silently be unavailable.
`requirements-opencv.txt` pins `>=4.8,<5`.

## Quick start (no weights, about a minute)

Try the page with fake detections first, to check your setup before downloading anything big:

```powershell
git clone https://github.com/BarisKucukkaya/ai-vision-demo.git
cd ai-vision-demo
python -m venv venv-gateway
venv-gateway\Scripts\python -m pip install -r requirements-opencv.txt -r requirements-dev.txt
$env:DEMO_MOCK = "1"; venv-gateway\Scripts\python -m uvicorn app.main:app --port 8000
```

bash: `python3 -m venv venv-gateway && venv-gateway/bin/python -m pip install -r requirements-opencv.txt -r requirements-dev.txt && DEMO_MOCK=1 venv-gateway/bin/python -m uvicorn app.main:app --port 8000`

Open <http://127.0.0.1:8000>, choose any photo with **Choose file** and press **Detect**. A yellow "Test model" banner confirms the results are fake.
For real models continue with the sections below.

## Install

Why three environments? TensorFlow and PyTorch are large, have their own native dependencies and release cadence.
Isolating them avoids version conflicts and lets the gateway start without either installed.

### Distributed setup (reference): three virtual environments

PowerShell (Windows):

```powershell
python -m venv venv-gateway; venv-gateway\Scripts\python -m pip install -r requirements-opencv.txt -r requirements-dev.txt
python -m venv venv-torch;   venv-torch\Scripts\python   -m pip install -r requirements-pytorch.txt -r requirements-dev.txt
python -m venv venv-tf;      venv-tf\Scripts\python      -m pip install -r requirements-tensorflow.txt -r requirements-dev.txt
```

bash (Linux / macOS):

```bash
python3 -m venv venv-gateway && venv-gateway/bin/python -m pip install -r requirements-opencv.txt -r requirements-dev.txt
python3 -m venv venv-torch   && venv-torch/bin/python   -m pip install -r requirements-pytorch.txt -r requirements-dev.txt
python3 -m venv venv-tf      && venv-tf/bin/python      -m pip install -r requirements-tensorflow.txt -r requirements-dev.txt
```

### All-in-one setup: a single environment

```powershell
python -m venv venv; venv\Scripts\python -m pip install -r requirements-opencv.txt -r requirements-pytorch.txt -r requirements-tensorflow.txt -r requirements-dev.txt
```

(bash: `python3 -m venv venv && venv/bin/python -m pip install -r requirements-opencv.txt -r requirements-pytorch.txt -r requirements-tensorflow.txt -r requirements-dev.txt`)

## Download the weights (once)

Weights go to the torchvision and KerasHub caches, never into the repo. HOG needs no download.

```powershell
venv-torch\Scripts\python scripts\download_weights.py --only retinanet-pytorch,faster-rcnn-pytorch
venv-tf\Scripts\python    scripts\download_weights.py --only retinanet-tensorflow
```

bash: `venv-torch/bin/python scripts/download_weights.py --only retinanet-pytorch,faster-rcnn-pytorch` and `venv-tf/bin/python scripts/download_weights.py --only retinanet-tensorflow`.
All-in-one: just `python scripts/download_weights.py`. The downloads are about 160 MB (RetinaNet), 170 MB (Faster R-CNN) and 140 MB (KerasHub RetinaNet) and
land in `~/.cache/torch` and the KerasHub cache. The script prints the measured parameter counts and the device; it exits non-zero if anything failed.

## Add your photos

Copy your photos into `samples/` (see `samples/README.txt`) and refresh the page. They are ignored by git, so **share photos with your team separately** (the repo only holds recorded boxes and scores for them, see below). The page also accepts a file picker, drag and drop and `Ctrl+V` paste, so it works with an empty `samples/` too.

## Run

### All-in-one (one process, every combo in-process)

```powershell
venv\Scripts\python -m uvicorn app.main:app --port 8000
```

bash: `venv/bin/python -m uvicorn app.main:app --port 8000`

### Distributed (a gateway plus one worker per framework)

One command (cross-platform), giving the interpreter of each environment:

```powershell
venv-gateway\Scripts\python scripts\run_distributed.py --gateway-python venv-gateway\Scripts\python.exe --torch-python venv-torch\Scripts\python.exe --tf-python venv-tf\Scripts\python.exe
```

```bash
venv-gateway/bin/python scripts/run_distributed.py --gateway-python venv-gateway/bin/python --torch-python venv-torch/bin/python --tf-python venv-tf/bin/python
```

Or by hand, in three terminals (PowerShell shown; in bash use `export VAR=value`):

```powershell
# worker 1: PyTorch models, port 8101
$env:DEMO_COMBOS = "faster_rcnn/pytorch,retinanet/pytorch"; venv-torch\Scripts\python -m uvicorn app.main:app --port 8101
# worker 2: TensorFlow model, port 8102
$env:DEMO_COMBOS = "retinanet/tensorflow"; venv-tf\Scripts\python -m uvicorn app.main:app --port 8102
# gateway: HOG locally, forwards the rest
$env:DEMO_COMBOS = "hog_svm/opencv"
$env:DEMO_REMOTES = "faster_rcnn/pytorch=http://127.0.0.1:8101,retinanet/pytorch=http://127.0.0.1:8101,retinanet/tensorflow=http://127.0.0.1:8102"
venv-gateway\Scripts\python -m uvicorn app.main:app --port 8000
```

The gateway needs neither torch nor tensorflow. Open <http://127.0.0.1:8000>. The status pills show which backends are ready (models load in the background).

### No weights? Mock and random-weight modes

* `DEMO_MOCK=1` (or `scripts/run_distributed.py --mock`): deterministic fake detections, no ML library needed.
* `DEMO_RANDOM_WEIGHTS=1`: the real code path with randomly initialised weights (meaningless output).

Both show a visible **test-model warning** in the page.

### Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `DEMO_COMBOS` | all four | combos run locally in this process |
| `DEMO_REMOTES` | empty | `combo=url` list forwarded to workers |
| `DEMO_MOCK` | 0 | 1 = every local adapter is a mock |
| `DEMO_RANDOM_WEIGHTS` | 0 | 1 = real code path, random weights |
| `DEMO_PRELOAD` | 1 | load adapters in a background thread at startup |
| `DEMO_DEVICE` | auto | `auto`, `cpu` or `cuda` for PyTorch |
| `TF_PRESET` | `retinanet_resnet50_fpn_coco` | KerasHub preset |
| `MAX_UPLOAD_MB` | 10 | upload size limit |
| `REMOTE_TIMEOUT_S` | 120 | worker request timeout |

## Test

```powershell
venv-gateway\Scripts\python -m pytest tests          # HOG, API, postprocess, remote, code endpoint (+ UI if Playwright is installed)
venv-torch\Scripts\python   -m pytest tests          # adds the torchvision tests
venv-tf\Scripts\python      -m pytest tests          # adds the KerasHub tests
node --test tests/js/geometry.test.mjs               # pure-JS geometry and tokenizer tests
```

Tests that need a library skip themselves when it is absent, so the suite passes in any single environment.
Browser checks (optional): `pip install playwright` and `python -m playwright install chromium` in `venv-gateway`; they run against `DEMO_MOCK=1`.

## Real-weights checks and recording (on the presenting machine)

```powershell
python scripts\smoke_test.py samples\street.jpg --expect person,car
python scripts\benchmark.py samples\street.jpg --runs 20 --json app\static\demo\data\benchmark.json
python scripts\count_lines.py --json app\static\demo\data\code_lines.json
python scripts\record_results.py
python scripts\snapshot_code.py
```

* `smoke_test.py` validates real results: boxes inside the image, expected classes found, and RetinaNet PyTorch vs TensorFlow agreeing (IoU >= 0.5). HOG is only required to find a person; a given photo may legitimately fail that.
* `record_results.py` writes `app/static/demo/data/recorded/<photo>.json` (boxes and scores only, never the photo): the offline **Recorded** fallback.
* `benchmark.json`, `code_lines.json`, `recorded/*.json` are picked up by the page automatically, no code change needed.
* Dress rehearsal on the laptop and projector you will use, **including a run with the network disabled**, and record the screen as the last-resort fallback.

## Troubleshooting

* **A pill says "loading" for a long time:** the models load in the background; the first start takes 20 to 60 seconds. A pill that turns to "unavailable" shows the reason when you hover it.
* **HOG + SVM unavailable:** you probably have OpenCV 5. Run `pip install "opencv-python-headless>=4.8,<5"` in that environment.
* **Weights cannot be downloaded:** check the internet connection and disk space, then run `download_weights.py` again. A failed load is retried after 30 seconds.
* **Port already in use:** the demo uses 8000 (and 8101, 8102 in distributed mode). Stop the other process or pass another `--port`.
* **Recorded mode shows boxes on a grey canvas:** the recorded data only has boxes and scores. Copy the matching photo into `samples/` to see them on the picture.
* **Real results look wrong:** run `scripts/smoke_test.py` on your photo; it checks boxes, classes and the PyTorch vs TensorFlow agreement.

## Layout

```
app/                 FastAPI app, adapters (hog_opencv, torchvision_detectors, keras_retinanet, remote, mock), static page
scripts/             download_weights, smoke_test, benchmark, record_results, snapshot_code, count_lines, run_distributed, train_cnn_*
tests/               Python tests, tests/js (node --test), tests/ui (Playwright)
samples/             your photos (not committed)
```

Keyboard: `D` Detect, `T` theme, `F` fullscreen for the results, `C` collapse or expand the code panel, `Esc` closes overlays (shortcuts are ignored while a field or slider has focus).

## Security note (read before putting it on the internet)

This is a presentation demo meant to run on your own machine. It has **no login and no rate limit**: anyone who can reach it can run models
(`/predict`, `/compare`: CPU cost) on images they send. The server never stores uploaded photos: a photo picked in the browser is analysed
and forgotten, and nothing can write into `samples/` through the web (you copy files there yourself). Uploads are validated (size limit, must be a readable image).
If you deploy it publicly, put it behind authentication or a rate-limiting reverse proxy, and do not publish photos you do not have the right to use.

## Honesty notes

No accuracy claim comes from this demo (no ground truth). Speeds are shown only from measured requests, with the device named.
Scores of different algorithms are not on the same scale. KerasHub's RetinaNet weights are not described as "separately trained":
differences between the two frameworks come from implementation details.
