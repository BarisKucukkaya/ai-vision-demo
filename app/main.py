"""FastAPI app: the page, the JSON API and background preloading."""
from __future__ import annotations

import io
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from . import codeexcerpts, config, registry
from .adapters.base import BackendError, BackendUnavailable
from .schemas import ImageInput, Options

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
HOG_STRIDES = (4, 8, 16)


def _preload(adapters) -> None:
    for adapter in adapters.values():
        try:
            adapter.ensure_loaded()
        except Exception:  # the error is cached on the adapter and shown in /health
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.adapters = registry.build_adapters()
    if config.preload():
        local = {c: a for c, a in app.state.adapters.items() if a.__class__.__name__ != "RemoteAdapter"}
        threading.Thread(target=_preload, args=(local,), daemon=True, name="preload").start()
    yield


app = FastAPI(title="AI algorithms compared", lifespan=lifespan)


# ---------------------------------------------------------------- helpers
def decode_upload(upload: UploadFile) -> ImageInput:
    limit = int(config.max_upload_mb() * 1024 * 1024)
    raw = upload.file.read(limit + 1)
    if len(raw) > limit:
        raise HTTPException(413, f"File is larger than {config.max_upload_mb():g} MB.")
    if not raw:
        raise HTTPException(400, "The uploaded file is empty.")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
        image = ImageOps.exif_transpose(image).convert("RGB")  # phones store rotation in EXIF; drop alpha
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError, SyntaxError):
        raise HTTPException(400, "The file is not a readable image.")
    return ImageInput(image=image, raw=raw, filename=upload.filename or "upload")


def make_options(nms_iou: float, hog_stride: int, hog_scale: float, hit_threshold: float) -> Options:
    if hog_stride not in HOG_STRIDES:
        raise HTTPException(422, f"hog_stride must be one of {list(HOG_STRIDES)}.")
    if not 1.01 <= hog_scale <= 1.5:
        raise HTTPException(422, "hog_scale must be between 1.01 and 1.5.")
    if not 0 <= hit_threshold <= 2:
        raise HTTPException(422, "hit_threshold must be between 0 and 2.")
    return Options(nms_iou=nms_iou, hog_stride=hog_stride, hog_scale=hog_scale, hit_threshold=hit_threshold)


def resolve_combo(algorithm: str, framework: str | None) -> str:
    valid = registry.valid_combos()
    if algorithm not in registry.ALGORITHMS:
        raise HTTPException(422, f"Unknown algorithm '{algorithm}'. Valid combinations: {valid}")
    if framework is None:
        framework = registry.ALGORITHMS[algorithm][0]
    if framework not in registry.KNOWN_FRAMEWORKS:
        raise HTTPException(422, f"Unknown framework '{framework}'. Valid combinations: {valid}")
    combo = f"{algorithm}/{framework}"
    if combo in registry.UNAVAILABLE:
        raise HTTPException(422, f"{combo} is not available: {registry.UNAVAILABLE[combo]}. Valid combinations: {valid}")
    if combo not in valid:
        raise HTTPException(422, f"Invalid combination '{combo}'. Valid combinations: {valid}")
    return combo


def get_adapter(combo: str):
    adapter = app.state.adapters.get(combo)
    if adapter is None:
        raise HTTPException(422, f"{combo} is not served by this process. Served here: {sorted(app.state.adapters)}")
    return adapter


# ---------------------------------------------------------------- API
@app.get("/health")
def health():
    adapters = app.state.adapters
    return {
        "status": "ok",
        "backends": {combo: a.info() for combo, a in adapters.items()},
        "algorithms": registry.ALGORITHMS,
        "unavailable": registry.UNAVAILABLE,
    }


@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    algorithm: str = "retinanet",
    framework: str | None = None,
    threshold: float = Query(0.5, ge=0, le=1),
    nms_iou: float = Query(0.5, ge=0.1, le=1.0),
    hog_stride: int = 8,
    hog_scale: float = 1.05,
    hit_threshold: float = 0.0,
):
    combo = resolve_combo(algorithm, framework)
    options = make_options(nms_iou, hog_stride, hog_scale, hit_threshold)
    adapter = get_adapter(combo)
    image = decode_upload(file)
    try:
        return adapter.predict(image, threshold, options).to_dict()
    except BackendUnavailable as exc:
        raise HTTPException(503, str(exc))
    except BackendError as exc:
        raise HTTPException(500, str(exc))


@app.post("/compare")
def compare(
    file: UploadFile = File(...),
    combos: str | None = None,
    threshold: float = Query(0.5, ge=0, le=1),
    nms_iou: float = Query(0.5, ge=0.1, le=1.0),
    hog_stride: int = 8,
    hog_scale: float = 1.05,
    hit_threshold: float = 0.0,
):
    options = make_options(nms_iou, hog_stride, hog_scale, hit_threshold)
    wanted = [c.strip() for c in combos.split(",") if c.strip()] if combos else list(app.state.adapters)
    for combo in wanted:  # unknown names fail the request, known-but-unavailable ones do not
        algorithm, framework = registry.split_combo(combo)
        if algorithm not in registry.ALGORITHMS or framework not in registry.KNOWN_FRAMEWORKS:
            raise HTTPException(422, f"Unknown combination '{combo}'. Valid combinations: {registry.valid_combos()}")
    image = decode_upload(file)
    results: dict[str, dict] = {}
    for combo in wanted:
        if combo in registry.UNAVAILABLE:
            results[combo] = {"ok": False, "unavailable": True, "error": registry.UNAVAILABLE[combo]}
            continue
        adapter = app.state.adapters.get(combo)
        if adapter is None:
            results[combo] = {"ok": False, "unavailable": True, "error": f"{combo} is not served by this process."}
            continue
        try:
            results[combo] = {"ok": True, **adapter.predict(image, threshold, options).to_dict()}
        except BackendUnavailable as exc:
            results[combo] = {"ok": False, "unavailable": True, "error": str(exc)}
        except Exception as exc:
            results[combo] = {"ok": False, "error": str(exc)}
    return {"image": {"width": image.width, "height": image.height}, "threshold": threshold, "results": results}


@app.get("/samples")
def samples():
    names: list[str] = []
    if config.SAMPLES_DIR.is_dir():
        names = sorted(p.name for p in config.SAMPLES_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    return {"samples": names}


@app.get("/samples/{name}")
def sample(name: str):
    root = config.SAMPLES_DIR.resolve()
    path = (root / name).resolve()
    if root not in path.parents or not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
        raise HTTPException(404, "Unknown sample.")
    return FileResponse(path)


@app.get("/api/code/{excerpt_id}")
def code(excerpt_id: str):
    result = codeexcerpts.extract(excerpt_id)
    if result is None:
        raise HTTPException(404, "Unknown code excerpt.")
    return result


@app.get("/api/data")
def optional_data():
    """Which optional data files exist, so the page never requests (and logs a 404 for) a missing one."""
    base = config.STATIC_DIR / "demo" / "data"
    recorded = base / "recorded"
    return {
        "files": sorted(p.name for p in base.glob("*.json")),
        "recorded": sorted(p.stem for p in recorded.glob("*.json")) if recorded.is_dir() else [],
    }


@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
