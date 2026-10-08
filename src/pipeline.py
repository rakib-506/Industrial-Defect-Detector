"""End-to-end inspection pipeline: detect, localise, then explain.

Stage 1 (PatchCore) decides whether a part is anomalous and where. Stage 2 (the
defect-type classifier) only runs when stage 1 says something is wrong - naming
a defect type on a clean part would be meaningless, since the classifier has
only ever seen defective examples.

Serving several categories at once, the expensive part is the ImageNet backbone,
which is identical across categories. `InspectionService` therefore keeps one
torch module per distinct PatchCore configuration and swaps only the memory bank
(the sole per-category state) when the requested category changes. Six separate
inferencers would duplicate a ~270 MB backbone six times, which does not fit
alongside the memory banks on a 4 GB card.
"""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image

from . import config, dataset, features
from .imaging import (
    anomaly_map_to_overlay,
    load_image,
    mask_to_outline,
    preprocess,
    to_data_uri,
)
from .patchcore_infer import _import_patchcore_model, _unpack


@dataclass
class InspectionResult:
    category: str
    anomaly_score: float
    threshold: float
    is_defective: bool
    severity: float
    defect_area_pct: float
    defect_type: str | None
    defect_probabilities: dict[str, float] | None
    images: dict[str, str]
    latency_ms: float
    # "pass" | "defect" | "uncertain". `is_defective` is kept alongside it so
    # existing consumers are unaffected; "uncertain" implies is_defective=True.
    verdict: str = "pass"
    message: str | None = None
    metrics: dict = field(default_factory=dict)


@dataclass
class CategoryAssets:
    category: str
    metrics: dict
    memory_bank: torch.Tensor  # kept on CPU until the category is activated
    settings: tuple  # (backbone, layers, num_neighbors) - the model signature
    classifier: object | None
    classes: list[str]

    @property
    def image_threshold(self) -> float:
        return float(self.metrics["image_threshold"])

    @property
    def pixel_threshold(self) -> float:
        return float(self.metrics["pixel_threshold"])

    @property
    def heatmap_range(self) -> tuple[float, float]:
        return tuple(self.metrics["heatmap_range"])

    @property
    def score_range(self) -> tuple[float, float]:
        return tuple(self.metrics["score_range"])

    @property
    def ood_threshold(self) -> float:
        """Anomalous-coverage fraction above which the product is not recognised."""
        return float(
            self.metrics.get(
                "out_of_distribution_threshold", config.OOD_COVERAGE_THRESHOLD
            )
        )


class InspectionService:
    """Serves every category that has been trained and calibrated."""

    def __init__(self, categories: list[str] | None = None, device: str | None = None):
        # DEFECT_DETECTOR_DEVICE lets the API run on CPU while the GPU is busy
        # training, and pins the device for deployment.
        device = device or os.environ.get("DEFECT_DETECTOR_DEVICE")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._assets: OrderedDict[str, CategoryAssets] = OrderedDict()
        self._models: dict[tuple, object] = {}
        self._active: tuple | None = None  # (signature, category) currently on device

        # How many memory banks to keep resident. Each is 110-150 MB, so caching
        # all seven costs roughly 1.8 GB of RSS versus 1.1 GB for one - the
        # difference between fitting on a small hosted instance and not.
        # Unlimited by default, so local behaviour is unchanged; set
        # DEFECT_DETECTOR_MAX_CACHED_BANKS=1 on a memory-constrained host, at the
        # cost of reloading ~130 MB from disk when the category changes.
        raw_cap = os.environ.get("DEFECT_DETECTOR_MAX_CACHED_BANKS", "").strip()
        self.max_cached_banks = int(raw_cap) if raw_cap.isdigit() and int(raw_cap) > 0 else None

        # Discovered from disk, so a newly onboarded category is picked up with
        # no code change and no manual registration.
        wanted = categories or config.trained_categories()
        self.categories = [c for c in wanted if self._is_ready(c)]
        if not self.categories:
            raise FileNotFoundError(
                "No trained categories found. Run "
                "`python -m src.run_all --categories <name> --nested-cv` first."
            )

    @staticmethod
    def _is_ready(category: str) -> bool:
        return config.patchcore_ckpt(category).exists() and config.metrics_path(category).exists()

    # -- lazy loading ------------------------------------------------------- #
    def _evict_if_needed(self, keep: str) -> None:
        """Drop least-recently-used memory banks once over the cap."""
        if self.max_cached_banks is None:
            return
        while len(self._assets) > self.max_cached_banks:
            oldest, _ = next(iter(self._assets.items()))
            if oldest == keep:  # never evict the category being served
                self._assets.move_to_end(oldest)
                continue
            self._assets.pop(oldest)
            # The evicted bank may also be the one currently on the device.
            if self._active is not None and self._active[1] == oldest:
                self._active = None

    def assets(self, category: str) -> CategoryAssets:
        if category in self._assets:
            self._assets.move_to_end(category)  # mark as recently used
        else:
            if category not in self.categories:
                raise KeyError(category)
            payload = torch.load(
                config.patchcore_ckpt(category), map_location="cpu", weights_only=False
            )
            metrics = json.loads(config.metrics_path(category).read_text())

            classifier = None
            classes: list[str] = []
            clf_path = config.classifier_path(category)
            if clf_path.exists():
                bundle = joblib.load(clf_path)
                classifier = bundle["pipeline"]
                classes = list(bundle["classes"])

            self._assets[category] = CategoryAssets(
                category=category,
                metrics=metrics,
                memory_bank=payload["memory_bank"].to(torch.float32),
                settings=(
                    payload["backbone"],
                    tuple(payload["layers"]),
                    payload["num_neighbors"],
                ),
                classifier=classifier,
                classes=classes,
            )
            self._evict_if_needed(keep=category)
        return self._assets[category]

    def _model_for(self, assets: CategoryAssets):
        """One torch module per distinct PatchCore configuration."""
        signature = assets.settings
        if signature not in self._models:
            backbone, layers, num_neighbors = signature
            model_cls = _import_patchcore_model()
            model = model_cls(
                backbone=backbone,
                layers=list(layers),
                pre_trained=True,
                num_neighbors=num_neighbors,
            )
            model.eval().to(self.device)
            self._models[signature] = model
        return self._models[signature]

    def _activate(self, assets: CategoryAssets):
        """Point the shared model at this category's memory bank."""
        model = self._model_for(assets)
        key = (assets.settings, assets.category)
        if self._active != key:
            model.memory_bank = assets.memory_bank.to(self.device)
            self._active = key
        return model

    # -- inference ---------------------------------------------------------- #
    @torch.no_grad()
    def inspect(
        self, category: str, source: str | Path | bytes | Image.Image
    ) -> InspectionResult:
        started = time.perf_counter()
        assets = self.assets(category)
        model = self._activate(assets)

        image = load_image(source)
        batch = preprocess(image).to(self.device)

        anomaly_map_t, pred_score = _unpack(model(batch))
        anomaly_map = anomaly_map_t.detach().cpu().float()
        if anomaly_map.dim() == 4:
            anomaly_map = anomaly_map.squeeze(1)
        anomaly_map = anomaly_map[0].numpy()
        score = float(pred_score.detach().cpu().float().flatten()[0])

        is_defective = score > assets.image_threshold
        defect_area_pct = float((anomaly_map >= assets.pixel_threshold).mean() * 100.0)
        # 0 at the threshold, 1 at twice the threshold - a readable "how bad".
        severity = float(
            np.clip(
                (score - assets.image_threshold) / max(assets.image_threshold, 1e-8), 0.0, 1.0
            )
        )

        # Safety check: is this even the right product? A genuine defect is
        # local, so most of a correct part still matches the memory bank. When
        # almost the whole frame is unfamiliar, the more likely explanation is
        # that this is a different object, and neither the threshold nor the
        # defect-type classifier means anything for it.
        coverage = defect_area_pct / 100.0
        out_of_distribution = is_defective and coverage >= assets.ood_threshold

        verdict = "pass"
        message: str | None = None
        if out_of_distribution:
            verdict = "uncertain"
            message = (
                f"This photo looks very different from a normal "
                f"{category.replace('_', ' ')} - it may not be the right product "
                f"for this checker. Please confirm you selected the correct product."
            )
        elif is_defective:
            verdict = "defect"

        defect_type: str | None = None
        probabilities: dict[str, float] | None = None
        # No defect type when the product is unrecognised: the classifier has
        # only ever seen defects of this product, so any label would be invented.
        if is_defective and not out_of_distribution and assets.classifier is not None:
            embedding = self._embed(model, batch)
            appearance, geometry = features.build_feature_vector(
                embedding, anomaly_map, assets.pixel_threshold
            )
            vector = np.concatenate([appearance, geometry]).astype(np.float64)[None]
            proba = assets.classifier.predict_proba(vector)[0]
            probabilities = {name: float(p) for name, p in zip(assets.classes, proba)}
            defect_type = max(probabilities, key=probabilities.get)

        overlay = anomaly_map_to_overlay(
            image, anomaly_map, vmin=assets.heatmap_range[0], vmax=assets.heatmap_range[1]
        )
        outline = mask_to_outline(image, anomaly_map, assets.pixel_threshold)

        return InspectionResult(
            category=category,
            anomaly_score=score,
            threshold=assets.image_threshold,
            is_defective=bool(is_defective),
            severity=severity,
            defect_area_pct=defect_area_pct,
            defect_type=defect_type,
            defect_probabilities=probabilities,
            images={
                "original": to_data_uri(image),
                "heatmap": to_data_uri(overlay),
                "outline": to_data_uri(outline),
            },
            latency_ms=(time.perf_counter() - started) * 1000.0,
            verdict=verdict,
            message=message,
            metrics=assets.metrics,
        )

    @staticmethod
    @torch.no_grad()
    def _embed(model, batch: torch.Tensor) -> torch.Tensor:
        """PatchCore's patch-embedding grid, (B, C, h, w)."""
        feats = model.feature_extractor(batch)
        feats = {layer: model.feature_pooler(f) for layer, f in feats.items()}
        return model.generate_embedding(feats).detach().cpu()

    # -- metadata ----------------------------------------------------------- #
    def summary(self, category: str) -> dict:
        assets = self.assets(category)
        m = assets.metrics
        clf = m.get("classifier") or {}
        return {
            "category": category,
            "type": config.category_type(category),
            "onboarded": config.is_onboarded(category),
            "has_classifier": assets.classifier is not None,
            "threshold": assets.image_threshold,
            "score_range": list(assets.score_range),
            "defect_classes": assets.classes or dataset.defect_labels(category),
            "metrics": {
                "image_auroc": m.get("image_auroc"),
                "pixel_auroc": m.get("pixel_auroc"),
                "image_average_precision": m.get("image_average_precision"),
                "pixel_f1_at_threshold": m.get("pixel_f1_at_threshold"),
                "at_threshold": m.get("at_threshold"),
                "counts": m.get("counts"),
                "classifier": clf,
            },
            "device": str(self.device),
        }


# Backwards-compatible single-category handle, used by scripts/smoke_test.py.
class InspectionPipeline:
    def __init__(self, category: str = config.DEFAULT_CATEGORY):
        self.category = category
        self.service = InspectionService([category])
        self.metrics = self.service.assets(category).metrics

    @property
    def device(self) -> str:
        return str(self.service.device)

    @property
    def image_threshold(self) -> float:
        return self.service.assets(self.category).image_threshold

    @property
    def classes(self) -> list[str]:
        return self.service.assets(self.category).classes

    def inspect(self, source) -> InspectionResult:
        return self.service.inspect(self.category, source)
