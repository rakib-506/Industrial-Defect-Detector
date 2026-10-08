"""FastAPI service exposing the inspection pipeline across MVTec categories.

Run:  python -m uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import io
import json
import sys
import traceback
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from .. import config
from ..samples import sample_gallery, sample_index
from ..pipeline import InspectionResult, InspectionService

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

_service: InspectionService | None = None
_startup_error: str | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm the service once at startup rather than on the first request, so the
    # first inspection a user runs is not misleadingly slow. Memory banks
    # themselves load lazily, per category, on first use, so this is cheap
    # (~0.2 s) and does not hold the port closed for long.
    #
    # A failure here must NOT stop the app from binding. On a hosted platform a
    # process that exits during startup is reported as a dead service with no
    # way to ask it why; staying up means /health answers and the error is
    # visible in the response and the logs.
    global _service, _startup_error
    try:
        _service = InspectionService()
        print(f"[api] ready on {_service.device}; categories: {', '.join(_service.categories)}")
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        _startup_error = f"{type(exc).__name__}: {exc}"
        print(f"[api] STARTUP FAILED: {_startup_error}", file=sys.stderr)
        traceback.print_exc()
    yield
    _service = None


app = FastAPI(
    title="Industrial Defect Detector",
    description="PatchCore anomaly detection with defect-type explainability, "
    "across six MVTec AD categories.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # Open to any origin: this is a public read-only demo, and the frontend may
    # be served from a different domain (e.g. Vercel) than the API. When the
    # container serves the UI itself, requests are same-origin and CORS never
    # applies, so this costs nothing there.
    #
    # allow_credentials stays False on purpose. The wildcard origin and
    # credentialed requests are mutually exclusive per the CORS spec, and no
    # endpoint here reads cookies or auth headers.
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", include_in_schema=False)
def liveness() -> dict:
    """Liveness probe for the hosting platform.

    Deliberately trivial: no model access, no disk access, no inference. It must
    answer while memory banks are still loading, and must keep answering even if
    startup failed - otherwise a platform health check reports the service dead
    without ever surfacing the reason.
    """
    return {
        "status": "ok",
        "models_ready": _service is not None,
        "startup_error": _startup_error,
    }

# Every endpoint lives under /api so the remaining paths can serve the built
# single-page app from the same origin and port - Hugging Face Spaces exposes
# exactly one port.
router = APIRouter()


def get_service() -> InspectionService:
    if _service is None:
        # Say which of the two it is, rather than blaming a slow load for a
        # configuration problem that will never resolve on its own.
        detail = (
            f"Model failed to load at startup: {_startup_error}"
            if _startup_error
            else "Model is still loading."
        )
        raise HTTPException(status_code=503, detail=detail)
    return _service


def resolve_category(service: InspectionService, category: str | None) -> str:
    if category is None:
        return (
            config.DEFAULT_CATEGORY
            if config.DEFAULT_CATEGORY in service.categories
            else service.categories[0]
        )
    if category not in service.categories:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown category '{category}'. Available: {', '.join(service.categories)}",
        )
    return category


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class PredictionOut(BaseModel):
    category: str
    # "pass" | "defect" | "uncertain". `is_defective` is unchanged for existing
    # consumers; "uncertain" always has is_defective=true.
    verdict: str
    message: str | None
    anomaly_score: float
    threshold: float
    is_defective: bool
    severity: float
    defect_type: str | None
    defect_probabilities: dict[str, float] | None
    defect_area_pct: float
    images: dict[str, str]
    latency_ms: float


class CategoryOut(BaseModel):
    category: str
    type: str
    onboarded: bool = False
    has_classifier: bool = True
    threshold: float
    score_range: list[float]
    defect_classes: list[str]
    metrics: dict
    device: str


class SampleOut(BaseModel):
    id: str
    category: str
    label: str
    thumbnail: str


def _to_out(result: InspectionResult) -> PredictionOut:
    return PredictionOut(
        category=result.category,
        verdict=result.verdict,
        message=result.message,
        anomaly_score=result.anomaly_score,
        threshold=result.threshold,
        is_defective=result.is_defective,
        severity=result.severity,
        defect_type=result.defect_type,
        defect_probabilities=result.defect_probabilities,
        defect_area_pct=result.defect_area_pct,
        images=result.images,
        latency_ms=result.latency_ms,
    )


# --------------------------------------------------------------------------- #
# Sample gallery
# --------------------------------------------------------------------------- #
# The gallery itself lives in src/samples.py so the Gradio Space serves exactly
# the same images and ids as this backend.
def _sample_index(category: str) -> dict:
    return sample_index(category)


def _sample_payload(category: str) -> list[SampleOut]:
    return [SampleOut(**entry) for entry in sample_gallery(category)]


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _service is not None,
        "categories": _service.categories if _service else [],
    }


@router.get("/categories", response_model=list[CategoryOut])
def categories() -> list[CategoryOut]:
    """Every trained category, with its own thresholds, classes and metrics."""
    service = get_service()
    return [CategoryOut(**service.summary(c)) for c in service.categories]


@router.get("/metadata", response_model=CategoryOut)
def metadata(category: str | None = Query(default=None)) -> CategoryOut:
    service = get_service()
    return CategoryOut(**service.summary(resolve_category(service, category)))


@router.get("/comparison")
def comparison() -> list[dict]:
    """The cross-category comparison table, as built by `python -m src.compare`."""
    if not config.COMPARISON_PATH.exists():
        raise HTTPException(
            status_code=404, detail="comparison.json not found - run `python -m src.compare`."
        )
    return json.loads(config.COMPARISON_PATH.read_text())


@router.get("/samples", response_model=list[SampleOut])
def samples(category: str | None = Query(default=None)) -> list[SampleOut]:
    service = get_service()
    try:
        return _sample_payload(resolve_category(service, category))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/predict", response_model=PredictionOut)
async def predict(
    file: UploadFile = File(...),
    category: str | None = Form(default=None),
) -> PredictionOut:
    service = get_service()
    resolved = resolve_category(service, category)

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image larger than 20 MB.")

    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Not a readable image file.") from exc

    return _to_out(service.inspect(resolved, image))


@router.post("/predict/sample/{sample_id}", response_model=PredictionOut)
def predict_sample(sample_id: str) -> PredictionOut:
    service = get_service()
    # Sample ids are "<category>__<label>__<stem>", so the category is implicit.
    category = sample_id.split("__", 1)[0]
    if category not in service.categories:
        raise HTTPException(status_code=404, detail=f"Unknown sample '{sample_id}'.")

    sample = _sample_index(category).get(sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail=f"Unknown sample '{sample_id}'.")
    return _to_out(service.inspect(category, sample.path))


# --------------------------------------------------------------------------- #
# Wiring. Routes first, then the static site, so /api never gets shadowed.
# --------------------------------------------------------------------------- #
app.include_router(router, prefix="/api")

FRONTEND_DIST = config.PROJECT_ROOT / "frontend" / "dist"

if FRONTEND_DIST.is_dir():
    # Hashed asset filenames, safe to cache hard.
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets"),
        name="assets",
    )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/{path:path}", include_in_schema=False)
    def spa_fallback(path: str) -> FileResponse:
        """Serve a real file when one exists, otherwise the SPA entry point."""
        candidate = (FRONTEND_DIST / path).resolve()
        # Keep the traversal inside dist; a crafted path must not escape it.
        if candidate.is_file() and candidate.is_relative_to(FRONTEND_DIST.resolve()):
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")

else:  # pragma: no cover - local development, UI served by Vite
    @app.get("/", include_in_schema=False)
    def index_missing() -> dict:
        return {
            "detail": "Frontend build not found. Run `npm run build` in frontend/, "
            "or use the Vite dev server on :5173.",
            "api": "/api",
            "docs": "/docs",
        }
