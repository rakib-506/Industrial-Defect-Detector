"""Gradio front door for Hugging Face Spaces.

A second, independent way to host the same detector. The Docker/Render path in
`Dockerfile` + `render.yaml` is unaffected by anything here.

What this is for: Hugging Face's free CPU tier has 16 GB of RAM, where Render's
free tier has 512 MB - and this service needs ~1.4 GB. So a Space can actually
run all seven categories, which the Render free plan cannot.

Nothing about detection is reimplemented. `InspectionService` is the same class
the FastAPI backend uses, so scores, thresholds, the wrong-product check and the
defect-type classifier all behave identically.

Two surfaces:
  * a small visible UI, so the Space is browsable
  * a named API endpoint `predict`, guarded by a shared secret, returning exactly
    the JSON shape `POST /api/predict` returns in the FastAPI backend

Run locally:
    set DEFECT_DETECTOR_API_KEY=dev-secret
    python -m deploy.gradio_app
"""

from __future__ import annotations

import hmac
import os
import sys
import time
from pathlib import Path

# Must be set before anything imports torch-backed modules. Spaces have no GPU.
os.environ.setdefault("DEFECT_DETECTOR_DEVICE", "cpu")
# One memory bank resident at a time. 16 GB is plenty for all seven, but holding
# only the active one keeps a cold Space responsive and memory flat.
os.environ.setdefault("DEFECT_DETECTOR_MAX_CACHED_BANKS", "1")

# Allow `python deploy/gradio_app.py` as well as `python -m deploy.gradio_app`:
# Hugging Face runs the app file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr  # noqa: E402
from PIL import Image  # noqa: E402

from src.pipeline import InspectionResult, InspectionService  # noqa: E402
from src.samples import resolve as resolve_sample  # noqa: E402
from src.samples import sample_gallery, sample_index  # noqa: E402

# --- ZeroGPU compatibility -------------------------------------------------- #
# Hugging Face's ZeroGPU hardware refuses to start a Space unless it finds at
# least one @spaces.GPU function ("No @spaces.GPU function detected during
# startup"). This app is deliberately CPU-only, so the requirement is met with a
# stub that nothing ever calls. Inference is untouched and still runs on CPU:
# DEFECT_DETECTOR_DEVICE is pinned above, and InspectionService short-circuits
# before it would ever query torch.cuda.
#
# The import is optional on purpose. `spaces` only exists on Hugging Face, and
# this same file has to keep working locally and under the Docker/Render
# deployment, where the package is absent.
try:  # pragma: no cover - depends on the host
    import spaces
except ImportError:
    spaces = None

if spaces is not None:

    @spaces.GPU(duration=1)
    def _zerogpu_startup_stub() -> str:
        """Exists only so ZeroGPU's startup scan finds a GPU function.

        Never called. Requests the smallest possible allocation so that even an
        accidental invocation costs nothing meaningful.
        """
        return "ok"

# Read from a Hugging Face Space secret of the same name. Unset means the API
# refuses every call: better a Space that is plainly misconfigured than one
# silently serving an open model endpoint.
API_KEY = os.environ.get("DEFECT_DETECTOR_API_KEY", "").strip()

_service: InspectionService | None = None


def service() -> InspectionService:
    """Load once, on first use, so the UI paints before the models are ready."""
    global _service
    if _service is None:
        _service = InspectionService()
        print(f"[gradio] ready on {_service.device}: {', '.join(_service.categories)}")
    return _service


def _categories() -> list[str]:
    try:
        return service().categories
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI instead of crashing
        print(f"[gradio] could not load models: {exc}")
        return []


def _as_payload(result: InspectionResult) -> dict:
    """Exactly the field set `POST /api/predict` returns.

    Kept identical on purpose: the frontend's result handling then works against
    either backend, and only the request URL differs.

    Types are coerced to plain Python here. The defect-class names originate from
    scikit-learn's `classes_`, which yields `numpy.str_` keys - pydantic silently
    accepts those in the FastAPI backend, but Gradio serialises with orjson,
    which rejects any non-`str` dict key outright.
    """
    probabilities = result.defect_probabilities
    if probabilities is not None:
        probabilities = {str(name): float(p) for name, p in probabilities.items()}

    return {
        "category": str(result.category),
        "verdict": str(result.verdict),
        "message": result.message,
        "anomaly_score": float(result.anomaly_score),
        "threshold": float(result.threshold),
        "is_defective": bool(result.is_defective),
        "severity": float(result.severity),
        "defect_type": None if result.defect_type is None else str(result.defect_type),
        "defect_probabilities": probabilities,
        "defect_area_pct": float(result.defect_area_pct),
        "images": {str(k): v for k, v in result.images.items()},
        "latency_ms": float(result.latency_ms),
    }


def _inspect(image: Image.Image, category: str) -> dict:
    """Shared core for both the UI and the API."""
    if image is None:
        return {"error": "No image supplied."}
    svc = service()
    if category not in svc.categories:
        return {
            "error": f"Unknown category '{category}'. "
            f"Available: {', '.join(svc.categories)}"
        }
    return _as_payload(svc.inspect(category, image))


# --------------------------------------------------------------------------- #
# API endpoint - shared secret required
# --------------------------------------------------------------------------- #
def predict_api(image: Image.Image, category: str, api_key: str = "") -> dict:
    """Authenticated endpoint. Exposed to clients as `predict`.

    The key travels as a parameter rather than a header because that is what
    Gradio's own call protocol carries. See the security note in the README: this
    keeps casual traffic off the endpoint, it is not real authentication.
    """
    if not API_KEY:
        return {
            "error": "Server misconfigured: DEFECT_DETECTOR_API_KEY is not set on "
            "this Space, so the API is disabled. Set it under Settings -> "
            "Variables and secrets."
        }
    # Constant-time compare, so response timing does not leak the key.
    if not hmac.compare_digest(str(api_key or ""), API_KEY):
        return {"error": "Invalid or missing API key."}

    started = time.perf_counter()
    payload = _inspect(image, category)
    if "error" not in payload:
        # Include queue/cold-start time the caller actually waited.
        payload["server_ms"] = (time.perf_counter() - started) * 1000.0
    return payload


def samples_api(category: str, api_key: str = "") -> dict:
    """Thumbnails for one category's demo gallery.

    Public visitors have no MVTec photographs of their own, so the gallery is
    what makes the demo usable at all. Served from the same `src.samples`
    helper the FastAPI backend uses, so ids and images cannot drift apart.
    """
    if not API_KEY:
        return {"error": "Server misconfigured: DEFECT_DETECTOR_API_KEY is not set."}
    if not hmac.compare_digest(str(api_key or ""), API_KEY):
        return {"error": "Invalid or missing API key."}

    svc = service()
    if category not in svc.categories:
        return {"error": f"Unknown category '{category}'."}
    return {"samples": sample_gallery(category)}


def predict_sample_api(sample_id: str, api_key: str = "") -> dict:
    """Inspect one bundled gallery image, addressed by its id.

    Sending the id rather than the image keeps a demo click cheap: the picture
    already lives on the server, so nothing has to be uploaded back to it.
    """
    if not API_KEY:
        return {"error": "Server misconfigured: DEFECT_DETECTOR_API_KEY is not set."}
    if not hmac.compare_digest(str(api_key or ""), API_KEY):
        return {"error": "Invalid or missing API key."}

    # Ids are "<category>__<label>__<stem>", so the category is implicit.
    category = str(sample_id).split("__", 1)[0]
    svc = service()
    if category not in svc.categories:
        return {"error": f"Unknown sample '{sample_id}'."}
    sample = resolve_sample(category, sample_id)
    if sample is None:
        return {"error": f"Unknown sample '{sample_id}'."}

    return _as_payload(svc.inspect(category, sample.path))


def metadata_api(api_key: str = "") -> dict:
    """Category list and metrics, matching `GET /api/categories`.

    The frontend needs this to populate its product switcher, so the Gradio
    backend has to answer it too or the UI comes up empty.
    """
    if not API_KEY:
        return {"error": "Server misconfigured: DEFECT_DETECTOR_API_KEY is not set."}
    if not hmac.compare_digest(str(api_key or ""), API_KEY):
        return {"error": "Invalid or missing API key."}

    svc = service()
    summaries = []
    for name in svc.categories:
        summary = svc.summary(name)
        # Same numpy-key coercion as _as_payload; see the note there.
        summary["defect_classes"] = [str(c) for c in summary.get("defect_classes", [])]
        summaries.append(summary)
    return {"categories": summaries}


# --------------------------------------------------------------------------- #
# Visible UI - open, since browsing the Space is the point of it
# --------------------------------------------------------------------------- #
def predict_ui(image: Image.Image, category: str):
    if image is None:
        return "Upload an image first.", None, None, {}

    payload = _inspect(image, category)
    if "error" in payload:
        return f"{payload['error']}", None, None, payload

    verdict = payload["verdict"]
    headline = {
        "pass": "PASS - looks normal",
        "defect": "DEFECT",
        "uncertain": "UNCERTAIN - may be the wrong product",
    }.get(verdict, verdict)

    lines = [
        f"## {headline}",
        "",
        f"- **Anomaly score:** {payload['anomaly_score']:.2f} "
        f"(threshold {payload['threshold']:.2f})",
        f"- **Affected area:** {payload['defect_area_pct']:.1f}% of the image",
    ]
    if verdict == "defect" and payload["defect_probabilities"]:
        best = max(payload["defect_probabilities"], key=payload["defect_probabilities"].get)
        confidence = payload["defect_probabilities"][best]
        lines.append(f"- **Likely defect type:** {best} ({confidence:.0%})")
    elif verdict == "uncertain":
        lines += ["", f"> {payload['message']}"]
    elif verdict == "defect":
        lines.append("- **Defect type:** not available for this category")

    def _decode(key: str) -> Image.Image | None:
        import base64
        import io

        uri = payload["images"].get(key)
        if not uri:
            return None
        return Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))

    return "\n".join(lines), _decode("heatmap"), _decode("outline"), payload


def build_demo() -> gr.Blocks:
    categories = _categories()
    default = "bottle" if "bottle" in categories else (categories[0] if categories else None)

    with gr.Blocks(title="Industrial Defect Detector") as demo:
        gr.Markdown(
            "# Industrial Defect Detector\n"
            "Upload a photo of a manufactured part and the detector decides whether "
            "it is faulty, shows where, and names the defect type.\n\n"
            "It learns only from **defect-free** examples, so it flags failure modes "
            "nobody labelled. **Pick the correct product first** - the result is "
            "meaningless against the wrong one, though a badly mismatched photo is "
            "usually caught and returned as `UNCERTAIN`."
        )

        with gr.Row():
            with gr.Column(scale=1):
                category = gr.Dropdown(
                    choices=categories,
                    value=default,
                    label="Product",
                    info="Must match what is in the photo.",
                )
                image = gr.Image(type="pil", label="Photo of the part")
                run = gr.Button("Inspect", variant="primary")
            with gr.Column(scale=1):
                summary = gr.Markdown(label="Result")
                with gr.Tab("Heatmap"):
                    heatmap = gr.Image(label="Where it looks abnormal")
                with gr.Tab("Outline"):
                    outline = gr.Image(label="Flagged region")
                with gr.Accordion("Raw JSON", open=False):
                    raw = gr.JSON(label="Same shape as POST /api/predict")

        # api_name=False keeps the open UI handler out of the published API, so
        # the documented endpoint is the authenticated one.
        run.click(
            predict_ui,
            inputs=[image, category],
            outputs=[summary, heatmap, outline, raw],
            api_name=False,
        )

        # Visitors arrive without photographs of their own, so give them one
        # passing and one failing part per category to click. Paths come from
        # the same bundled gallery the API serves.
        example_rows: list[list[str]] = []
        for name in categories:
            found = sample_index(name)
            good = next((s for s in found.values() if not s.is_defective), None)
            bad = next((s for s in found.values() if s.is_defective), None)
            for chosen in (good, bad):
                if chosen is not None:
                    example_rows.append([str(chosen.path), name])

        if example_rows:
            gr.Examples(
                examples=example_rows,
                inputs=[image, category],
                outputs=[summary, heatmap, outline, raw],
                fn=predict_ui,
                # Inspect straight away, but do not pre-compute every example at
                # startup - that would delay a cold Space by a dozen inferences.
                run_on_click=True,
                cache_examples=False,
                label="Example parts - click one to inspect it",
                examples_per_page=14,
            )

        # The API surface. Hidden from the UI; this is what clients call.
        with gr.Row(visible=False):
            api_image = gr.Image(type="pil")
            api_category = gr.Textbox()
            api_key = gr.Textbox()
            api_out = gr.JSON()
            gr.Button().click(
                predict_api,
                inputs=[api_image, api_category, api_key],
                outputs=[api_out],
                api_name="predict",
            )

            meta_key = gr.Textbox()
            meta_out = gr.JSON()
            gr.Button().click(
                metadata_api,
                inputs=[meta_key],
                outputs=[meta_out],
                api_name="metadata",
            )

            samp_cat = gr.Textbox()
            samp_key = gr.Textbox()
            samp_out = gr.JSON()
            gr.Button().click(
                samples_api,
                inputs=[samp_cat, samp_key],
                outputs=[samp_out],
                api_name="samples",
            )

            ps_id = gr.Textbox()
            ps_key = gr.Textbox()
            ps_out = gr.JSON()
            gr.Button().click(
                predict_sample_api,
                inputs=[ps_id, ps_key],
                outputs=[ps_out],
                api_name="predict_sample",
            )

        gr.Markdown(
            "---\n"
            "Trained on the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) "
            "(CC BY-NC-SA 4.0). Detection uses PatchCore via anomalib; a small "
            "classifier names the defect type. Non-commercial use only."
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    if not API_KEY:
        print(
            "[gradio] WARNING: DEFECT_DETECTOR_API_KEY is not set. The UI works, "
            "but the `predict` API will refuse every call.",
            file=sys.stderr,
        )
    demo.queue(max_size=8).launch(
        server_name=os.environ.get("HOST", "0.0.0.0"),  # noqa: S104 - container
        server_port=int(os.environ.get("PORT", "7860")),
        show_api=True,
    )
