"""Standalone PatchCore inference.

Training happens through anomalib, but the only stateful thing PatchCore
produces is a coreset memory bank of patch embeddings. We persist that bank and
reload it into anomalib's ``PatchcoreModel`` torch module here, which keeps
inference numerically identical to training-time evaluation while avoiding
Lightning checkpoints and the ``Engine`` at serving time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import config


def _import_patchcore_model():
    """Locate anomalib's PatchcoreModel across the 1.x / 2.x layouts."""
    candidates = (
        "anomalib.models.image.patchcore.torch_model",
        "anomalib.models.patchcore.torch_model",
    )
    errors = []
    for module_path in candidates:
        try:
            module = __import__(module_path, fromlist=["PatchcoreModel"])
            return module.PatchcoreModel
        except ImportError as exc:  # pragma: no cover - depends on install
            errors.append(f"{module_path}: {exc}")
    raise ImportError(
        "Could not import PatchcoreModel from anomalib. Tried:\n  "
        + "\n  ".join(errors)
    )


def _unpack(output) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalise the model output across anomalib versions.

    Returns ``(anomaly_map, pred_score)``.
    """
    if hasattr(output, "anomaly_map") and hasattr(output, "pred_score"):
        return output.anomaly_map, output.pred_score
    if isinstance(output, dict):
        return output["anomaly_map"], output["pred_score"]
    if isinstance(output, (tuple, list)) and len(output) == 2:
        return output[0], output[1]
    raise TypeError(f"Unexpected PatchcoreModel output type: {type(output)!r}")


@dataclass
class PatchCoreResult:
    """One image's raw PatchCore output, before any thresholding."""

    score: float
    anomaly_map: np.ndarray  # (H, W) float32, raw distances


class PatchCoreInferencer:
    def __init__(
        self,
        category: str = config.DEFAULT_CATEGORY,
        checkpoint_path: Path | None = None,
        device: str | None = None,
    ):
        self.category = category
        checkpoint_path = Path(checkpoint_path or config.patchcore_ckpt(category))
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"PatchCore memory bank not found at {checkpoint_path}. "
                f"Run `python -m src.train_patchcore --category {category}` first."
            )

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        model_cls = _import_patchcore_model()
        self.model = model_cls(
            backbone=payload["backbone"],
            layers=payload["layers"],
            pre_trained=True,
            num_neighbors=payload["num_neighbors"],
        )
        memory_bank = payload["memory_bank"].to(torch.float32)
        # ``memory_bank`` is a dynamically sized buffer; assign rather than
        # load_state_dict so the shape is adopted as-is.
        self.model.memory_bank = memory_bank
        self.model.eval().to(self.device)
        self.model.memory_bank = self.model.memory_bank.to(self.device)

        self.image_size = tuple(payload.get("image_size", config.IMAGE_SIZE))

    @torch.no_grad()
    def predict_batch(self, batch: torch.Tensor) -> list[PatchCoreResult]:
        """Run a preprocessed (B, 3, H, W) batch through PatchCore."""
        batch = batch.to(self.device)
        anomaly_map, pred_score = _unpack(self.model(batch))
        anomaly_map = anomaly_map.detach().cpu().float()
        if anomaly_map.dim() == 4:  # (B, 1, H, W)
            anomaly_map = anomaly_map.squeeze(1)
        scores = pred_score.detach().cpu().float().flatten()
        return [
            PatchCoreResult(score=float(scores[i]), anomaly_map=anomaly_map[i].numpy())
            for i in range(anomaly_map.shape[0])
        ]

    def predict(self, batch: torch.Tensor) -> PatchCoreResult:
        return self.predict_batch(batch)[0]

    @torch.no_grad()
    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        """Return PatchCore's patch embedding grid for a batch.

        Shape ``(B, C, h, w)``. Used by the defect-type classifier so it reasons
        over the same representation the detector uses.
        """
        batch = batch.to(self.device)
        features = self.model.feature_extractor(batch)
        features = {
            layer: self.model.feature_pooler(feature)
            for layer, feature in features.items()
        }
        embedding = self.model.generate_embedding(features)
        return embedding.detach().cpu()
