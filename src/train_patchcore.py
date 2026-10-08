"""Train the PatchCore anomaly detector for one MVTec category.

PatchCore is trained on defect-free images only: it builds a coreset-subsampled
memory bank of patch embeddings from `train/good`, then scores a new image by
the distance of its patches to that bank. Nothing here sees a defect label.

Run:  python -m src.train_patchcore --category bottle
"""

from __future__ import annotations

import argparse
import json
import warnings

import torch

warnings.filterwarnings("ignore", category=FutureWarning)

from anomalib.data import MVTecAD  # noqa: E402
from anomalib.data.utils.split import ValSplitMode  # noqa: E402
from anomalib.engine import Engine  # noqa: E402
from anomalib.models import Patchcore  # noqa: E402

from . import config  # noqa: E402


def build_datamodule(category: str, batch_size: int) -> MVTecAD:
    return MVTecAD(
        # Onboarded categories live under a different root but the same layout.
        root=config.dataset_root_for(category),
        category=category,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        # Windows + Lightning: worker processes re-import and can deadlock on
        # small datasets, and loading a few hundred PNGs is not the bottleneck.
        num_workers=0,
        # Carve the validation set out of train/good instead of anomalib's
        # default of reusing the test set. Training images land in the memory
        # bank and therefore self-match at ~0 distance, so they cannot calibrate
        # a threshold; these held-out good images can, without touching test.
        val_split_mode=ValSplitMode.FROM_TRAIN,
        val_split_ratio=config.VAL_SPLIT_RATIO,
        seed=config.RANDOM_SEED,
    )


def train_category(
    category: str,
    batch_size: int = 8,
    accelerator: str = "auto",
    overrides: dict | None = None,
) -> dict:
    """Fit PatchCore for one category and persist its memory bank."""
    settings = config.patchcore_settings(category)
    if overrides:
        settings.update(overrides)

    config.models_dir(category).mkdir(parents=True, exist_ok=True)
    config.outputs_dir(category).mkdir(parents=True, exist_ok=True)

    datamodule = build_datamodule(category, batch_size)

    model = Patchcore(
        backbone=settings["backbone"],
        layers=settings["layers"],
        pre_trained=True,
        coreset_sampling_ratio=settings["coreset_ratio"],
        num_neighbors=settings["num_neighbors"],
        # Explicit so the serving path can mirror it exactly: resize to 256 and
        # normalise, with no centre crop that could clip an edge defect.
        pre_processor=Patchcore.configure_pre_processor(image_size=config.IMAGE_SIZE),
        # We render our own overlays in `imaging.py`; anomalib's visualiser would
        # otherwise dump a PNG per test image into outputs/ for every category.
        visualizer=False,
    )

    engine = Engine(
        default_root_dir=config.outputs_dir(category) / "anomalib",
        accelerator=accelerator,
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=True,
    )

    print(f"[train:{category}] fitting PatchCore (train/good only), settings={settings}")
    engine.fit(model=model, datamodule=datamodule)

    print(f"[train:{category}] evaluating on the labelled test split…")
    test_results = engine.test(model=model, datamodule=datamodule)

    memory_bank = model.model.memory_bank.detach().cpu().to(torch.float32)
    print(f"[train:{category}] memory bank: {tuple(memory_bank.shape)}")

    # Persist which good images were held out of the memory bank; calibration
    # scores exactly these to pick a threshold without seeing test labels.
    val_paths = [str(p) for p in datamodule.val_data.samples["image_path"]]
    train_paths = [str(p) for p in datamodule.train_data.samples["image_path"]]
    config.splits_path(category).write_text(
        json.dumps({"val_good": val_paths, "train_good": train_paths}, indent=2)
    )
    print(
        f"[train:{category}] {len(train_paths)} images in bank, "
        f"{len(val_paths)} good images held out for calibration"
    )

    torch.save(
        {
            "category": category,
            "backbone": settings["backbone"],
            "layers": list(settings["layers"]),
            "num_neighbors": settings["num_neighbors"],
            "coreset_ratio": settings["coreset_ratio"],
            "image_size": list(config.IMAGE_SIZE),
            "memory_bank": memory_bank,
        },
        config.patchcore_ckpt(category),
    )
    print(f"[train:{category}] saved memory bank -> {config.patchcore_ckpt(category)}")

    raw = test_results[0] if test_results else {}
    metrics = {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}
    summary_path = config.outputs_dir(category) / "patchcore_test_metrics.json"
    summary_path.write_text(json.dumps(metrics, indent=2))
    for key, value in metrics.items():
        print(f"         {key}: {value:.4f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default=config.DEFAULT_CATEGORY)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--accelerator",
        default="auto",
        help="'gpu', 'cpu' or 'auto'. Fall back to cpu if the GPU runs out of memory.",
    )
    parser.add_argument("--layers", nargs="+", default=None, help="Override PatchCore layers.")
    parser.add_argument("--backbone", default=None, help="Override the backbone.")
    parser.add_argument("--coreset-ratio", type=float, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.layers:
        overrides["layers"] = args.layers
    if args.backbone:
        overrides["backbone"] = args.backbone
    if args.coreset_ratio is not None:
        overrides["coreset_ratio"] = args.coreset_ratio

    train_category(args.category, args.batch_size, args.accelerator, overrides or None)


if __name__ == "__main__":
    main()
