"""Enumerate the MVTec category folders on disk.

Defect classes are always discovered from the `test/` subfolders, never
hardcoded - categories carry anywhere from 3 to 5 of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import config

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str  # "good" or a defect-type folder name
    category: str
    mask_path: Path | None

    @property
    def is_defective(self) -> bool:
        return self.label != "good"

    @property
    def sample_id(self) -> str:
        return f"{self.category}__{self.label}__{self.path.stem}"


def _mask_for(category: str, image_path: Path, label: str) -> Path | None:
    """MVTec stores masks as ground_truth/<label>/<stem>_mask.png."""
    if label == "good":
        return None
    candidate = (
        config.category_dir(category) / "ground_truth" / label / f"{image_path.stem}_mask.png"
    )
    return candidate if candidate.exists() else None


def list_split(category: str, split: str) -> list[Sample]:
    """List every image in `test` or `train`, sorted for reproducibility."""
    root = config.category_dir(category) / split
    if not root.exists():
        raise FileNotFoundError(f"Missing split directory: {root}")

    samples: list[Sample] = []
    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = label_dir.name
        for image_path in sorted(label_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append(
                    Sample(
                        path=image_path,
                        label=label,
                        category=category,
                        mask_path=_mask_for(category, image_path, label),
                    )
                )
    return samples


def load_mask(sample: Sample, size: tuple[int, int]) -> np.ndarray:
    """Binary ground-truth mask resized to ``size`` (H, W). Good parts -> zeros."""
    if sample.mask_path is None:
        return np.zeros(size, dtype=np.uint8)
    mask = Image.open(sample.mask_path).convert("L").resize(size[::-1], Image.NEAREST)
    return (np.asarray(mask) > 127).astype(np.uint8)


def has_split(category: str, split: str) -> bool:
    """Whether a split directory exists and holds at least one image.

    An onboarded category with no labelled defects has no `test/` at all, so
    every consumer must check before assuming one.
    """
    root = config.category_dir(category) / split
    if not root.is_dir():
        return False
    return any(
        p.suffix.lower() in IMAGE_SUFFIXES
        for d in root.iterdir()
        if d.is_dir()
        for p in d.iterdir()
    )


def defect_labels(category: str) -> list[str]:
    """Defect-type folder names present in the test split, excluding 'good'."""
    if not has_split(category, "test"):
        return []
    return sorted({s.label for s in list_split(category, "test") if s.is_defective})


def defect_counts(category: str) -> dict[str, int]:
    """How many labelled images each defect type has."""
    if not has_split(category, "test"):
        return {}
    counts: dict[str, int] = {}
    for sample in list_split(category, "test"):
        if sample.is_defective:
            counts[sample.label] = counts.get(sample.label, 0) + 1
    return dict(sorted(counts.items()))


def available_categories() -> list[str]:
    """Categories that exist on disk with the expected MVTec layout."""
    if not config.DATASET_ROOT.exists():
        return []
    found = []
    for path in sorted(config.DATASET_ROOT.iterdir()):
        if path.is_dir() and (path / "train" / "good").is_dir() and (path / "test").is_dir():
            found.append(path.name)
    return found
