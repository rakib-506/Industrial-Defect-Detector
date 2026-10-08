"""The demo sample gallery, shared by both backends.

Public visitors do not arrive with MVTec photographs of their own, so every
deployment needs a set of example images to click. The FastAPI backend and the
Gradio Space both serve the same gallery from this module, so the two cannot
drift apart.

Images resolve through the normal dataset roots, which means a deployment falls
back to the small bundled set in `deploy/samples/` when the raw dataset is not
present - exactly the situation inside a container.
"""

from __future__ import annotations

from functools import lru_cache

from PIL import Image

from . import dataset
from .imaging import to_data_uri

# Per defect type, per category. Two keeps the gallery scannable and the payload
# small; every category still leads with a passing part.
SAMPLES_PER_LABEL = 2
THUMBNAIL_PX = 160


@lru_cache(maxsize=None)
def sample_index(category: str) -> dict[str, dataset.Sample]:
    """Sample id -> Sample, class-balanced, 'good' first.

    Empty for a category with no test split - an onboarded category supplied
    without labelled defects simply has no gallery, and upload still works.
    """
    if not dataset.has_split(category, "test"):
        return {}

    per_label: dict[str, list] = {}
    for sample in dataset.list_split(category, "test"):
        per_label.setdefault(sample.label, []).append(sample)

    chosen: dict[str, dataset.Sample] = {}
    # 'good' first so the gallery opens with a part that passes.
    ordered = ["good"] + [label for label in sorted(per_label) if label != "good"]
    for label in ordered:
        for sample in per_label.get(label, [])[:SAMPLES_PER_LABEL]:
            chosen[sample.sample_id] = sample
    return chosen


@lru_cache(maxsize=None)
def sample_gallery(category: str) -> list[dict]:
    """Thumbnails for one category, ready to serve as JSON.

    Cached because encoding a dozen JPEGs per category is pure repeat work.
    """
    gallery: list[dict] = []
    for sample_id, sample in sample_index(category).items():
        thumb = Image.open(sample.path).convert("RGB")
        thumb.thumbnail((THUMBNAIL_PX, THUMBNAIL_PX), Image.LANCZOS)
        gallery.append(
            {
                "id": sample_id,
                "category": str(sample.category),
                "label": str(sample.label),
                "thumbnail": to_data_uri(thumb, "JPEG"),
            }
        )
    return gallery


def resolve(category: str, sample_id: str) -> dataset.Sample | None:
    """Look up one gallery sample, or None if the id is unknown."""
    return sample_index(category).get(sample_id)
