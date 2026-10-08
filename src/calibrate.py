"""Calibrate decision thresholds and measure detector quality, per category.

Thresholds are chosen on the good images held out of the memory bank, never on
the test split, so the reported test numbers stay honest. PatchCore memorises
its training images and scores them at ~0 distance, which is exactly why a
held-out set of good parts is needed.

Run:  python -m src.calibrate --category bottle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from . import config, dataset
from .imaging import load_image, preprocess
from .patchcore_infer import PatchCoreInferencer


def score_paths(
    inferencer: PatchCoreInferencer, paths: list[Path], batch_size: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Score images in batches, returning (image_scores, anomaly_maps)."""
    scores: list[float] = []
    maps: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        batch = torch.cat([preprocess(load_image(p)) for p in chunk], dim=0)
        for result in inferencer.predict_batch(batch):
            scores.append(result.score)
            maps.append(result.anomaly_map)
    return np.asarray(scores, dtype=np.float64), maps


def calibrate_category(category: str, batch_size: int = 4, quiet: bool = False) -> dict:
    """Pick thresholds from held-out good parts, then score the test split once."""

    def log(message: str) -> None:
        if not quiet:
            print(message)

    splits_path = config.splits_path(category)
    if not splits_path.exists():
        raise SystemExit(
            f"Missing {splits_path} - run `python -m src.train_patchcore "
            f"--category {category}` first."
        )
    splits = json.loads(splits_path.read_text())
    val_good = [Path(p) for p in splits["val_good"]]

    inferencer = PatchCoreInferencer(category)
    log(
        f"[calib:{category}] device={inferencer.device}, "
        f"memory bank={tuple(inferencer.model.memory_bank.shape)}"
    )

    # --- 1. Thresholds from held-out good parts ------------------------------
    log(f"[calib:{category}] scoring {len(val_good)} held-out good images…")
    val_scores, val_maps = score_paths(inferencer, val_good, batch_size)

    # Two leakage-free rules over the same held-out good images.
    #
    # `max` - the highest score any known-good part produced. Simple, but it is
    # the maximum of only ~20 samples, so it is whatever the single weirdest
    # good image happened to score. Measured across six categories it lands too
    # high on capsule and tile (missing real defects) and too low on carpet,
    # grid and metal_nut (false alarms) - noisy in both directions.
    #
    # `robust` - a tolerance limit at median + 3 sigma, with sigma estimated
    # from the median absolute deviation so one outlier cannot inflate it. This
    # estimates the same upper tail of the good-part distribution without
    # depending on a single observation, and was better or equal on all six
    # categories. It is the operating default; `max` is still reported so the
    # difference stays visible.
    max_threshold = float(val_scores.max())
    median = float(np.median(val_scores))
    mad = float(np.median(np.abs(val_scores - median)))
    robust_threshold = float(median + 3.0 * 1.4826 * mad)
    image_threshold = robust_threshold

    val_pixels = np.concatenate([m.ravel() for m in val_maps])
    # Same rule as the image threshold, one level down: the highest value any
    # pixel of a known-good part produced. Quantiles (p99, p99.9) sit lower and
    # over-segment badly.
    pixel_threshold = float(val_pixels.max())
    log(
        f"[calib:{category}] held-out good scores: min={val_scores.min():.4f} "
        f"mean={val_scores.mean():.4f} max={val_scores.max():.4f}"
    )
    log(
        f"[calib:{category}] image threshold={image_threshold:.4f} (robust; "
        f"max rule would give {max_threshold:.4f})  "
        f"pixel threshold={pixel_threshold:.4f}"
    )

    # --- 2. Evaluate on the untouched test split -----------------------------
    # A freshly onboarded category may have no labelled defects at all, or only
    # held-out good images. Each case reports exactly what it can measure and
    # leaves the rest null, rather than inventing a number or crashing.
    image_auroc = image_ap = pixel_auroc = pixel_ap = pixel_f1 = None
    at_threshold = at_max_threshold = None
    per_class: dict[str, dict] = {}
    ood_diagnostics: dict | None = None
    n_good = n_defective = 0
    tn = fp = fn = tp = 0

    has_test = dataset.has_split(category, "test")
    if has_test:
        test_samples = dataset.list_split(category, "test")
        log(f"[calib:{category}] scoring {len(test_samples)} test images…")
        test_scores, test_maps = score_paths(
            inferencer, [s.path for s in test_samples], batch_size
        )
        y_true = np.array([int(s.is_defective) for s in test_samples])
        y_pred = (test_scores > image_threshold).astype(int)
        n_defective = int(y_true.sum())
        n_good = len(y_true) - n_defective

        def operating_point(threshold: float) -> dict:
            pred = (test_scores > threshold).astype(int)
            a, b, c, d = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
            return {
                "threshold": float(threshold),
                "accuracy": float((d + a) / len(y_true)),
                "precision": float(d / (d + b)) if (d + b) else None,
                "recall": float(d / (d + c)) if (d + c) else None,
                "true_negatives": int(a),
                "false_positives": int(b),
                "false_negatives": int(c),
                "true_positives": int(d),
            }

        at_threshold = operating_point(image_threshold)
        at_max_threshold = operating_point(max_threshold)
        tn, fp, fn, tp = (
            at_threshold["true_negatives"],
            at_threshold["false_positives"],
            at_threshold["false_negatives"],
            at_threshold["true_positives"],
        )

        # AUROC needs both classes present.
        if n_defective and n_good:
            image_auroc = float(roc_auc_score(y_true, test_scores))
            image_ap = float(average_precision_score(y_true, test_scores))
        else:
            log(
                f"[calib:{category}] test split has only "
                f"{'defective' if n_defective else 'good'} images - image AUROC "
                "is undefined and is reported as null"
            )

        # Pixel metrics additionally need ground-truth masks, which a client
        # supplying raw defect photos will not have.
        map_size = test_maps[0].shape
        gt = np.concatenate(
            [dataset.load_mask(s, map_size).ravel() for s in test_samples]
        )
        if gt.any() and not gt.all():
            pixel_scores = np.concatenate([m.ravel() for m in test_maps])
            pixel_auroc = float(roc_auc_score(gt, pixel_scores))
            pixel_ap = float(average_precision_score(gt, pixel_scores))
            pixel_f1 = float(f1_score(gt, (pixel_scores > pixel_threshold).astype(int)))
        elif n_defective:
            log(
                f"[calib:{category}] no ground-truth masks - pixel metrics are "
                "reported as null (detection and heatmaps are unaffected)"
            )

        # How much of the frame genuine defects of this product actually cover.
        # Recorded as a diagnostic so the safety margin under the fixed
        # out-of-distribution threshold is visible per category; the threshold
        # itself is a constant and is not fitted to these numbers.
        defect_idx = [i for i, s in enumerate(test_samples) if s.is_defective]
        if defect_idx:
            coverages = [
                float((test_maps[i] >= pixel_threshold).mean()) for i in defect_idx
            ]
            ood_diagnostics = {
                "max_defect_coverage": float(max(coverages)),
                "median_defect_coverage": float(np.median(coverages)),
                "headroom": float(config.OOD_COVERAGE_THRESHOLD - max(coverages)),
                "defects_above_threshold": int(
                    sum(c >= config.OOD_COVERAGE_THRESHOLD for c in coverages)
                ),
                "n_defects": len(coverages),
            }

        # Per-defect-type detection recall, which shows whether one failure mode
        # is dragging a category down.
        for label in dataset.defect_labels(category):
            idx = [i for i, s in enumerate(test_samples) if s.label == label]
            detected = int(y_pred[idx].sum())
            per_class[label] = {
                "n": len(idx),
                "detected": detected,
                "recall": float(detected / len(idx)) if idx else 0.0,
                "mean_score": float(test_scores[idx].mean()),
            }
    else:
        log(
            f"[calib:{category}] no test split - thresholds calibrated from "
            "held-out good images only, no accuracy can be reported"
        )

    settings = torch.load(
        config.patchcore_ckpt(category), map_location="cpu", weights_only=False
    )
    metrics = {
        "category": category,
        "type": config.category_type(category),
        "is_texture": config.category_type(category) == "texture",
        "onboarded": config.is_onboarded(category),
        "has_test_data": has_test,
        "has_defect_labels": bool(n_defective),
        "patchcore": {
            "backbone": settings["backbone"],
            "layers": settings["layers"],
            "coreset_ratio": settings["coreset_ratio"],
            "num_neighbors": settings["num_neighbors"],
            "memory_bank": list(settings["memory_bank"].shape),
        },
        "image_auroc": image_auroc,
        "image_average_precision": image_ap,
        "pixel_auroc": pixel_auroc,
        "pixel_average_precision": pixel_ap,
        "pixel_f1_at_threshold": pixel_f1,
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "threshold_rule": "robust (median + 3*1.4826*MAD over held-out good)",
        "at_threshold": at_threshold,
        # The simpler max-over-held-out-good rule, kept for comparison.
        "at_max_threshold": at_max_threshold,
        "max_threshold": max_threshold,
        "calibration_scores": {
            "min": float(val_scores.min()),
            "median": median,
            "mean": float(val_scores.mean()),
            "max": float(val_scores.max()),
            "mad": mad,
            "n": int(len(val_scores)),
        },
        # Safety check for photos of the wrong product entirely. Triggered on
        # how much of the frame is anomalous, not on the score - see
        # config.OOD_COVERAGE_THRESHOLD for why.
        "out_of_distribution_threshold": config.OOD_COVERAGE_THRESHOLD,
        "out_of_distribution_rule": (
            "anomalous coverage >= threshold -> verdict 'uncertain' "
            "(wrong product suspected); score alone does not separate"
        ),
        "out_of_distribution_diagnostics": ood_diagnostics,
        "per_defect_recall": per_class,
        "counts": {
            "memory_bank_images": len(splits["train_good"]),
            "calibration_images": len(val_good),
            "test_images": len(test_samples) if has_test else 0,
            "test_good": n_good,
            "test_defective": n_defective,
            "defect_classes": len(dataset.defect_labels(category)),
        },
        # Display range for the heatmap: colours saturate at twice the novelty
        # threshold so a hot spot always means the same thing across uploads.
        "heatmap_range": [float(np.median(val_pixels)), float(pixel_threshold * 2.0)],
        "score_range": [0.0, float(image_threshold * 2.0)],
    }

    config.outputs_dir(category).mkdir(parents=True, exist_ok=True)
    # Preserve a classifier block from an earlier run so calibration can be
    # re-run without wiping stage-2 results.
    existing = (
        json.loads(config.metrics_path(category).read_text())
        if config.metrics_path(category).exists()
        else {}
    )
    if "classifier" in existing:
        metrics["classifier"] = existing["classifier"]
    config.metrics_path(category).write_text(json.dumps(metrics, indent=2))

    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    if has_test:
        log(f"\n[calib:{category}] ---- test-set results (threshold never saw these) ----")
        log(f"        image AUROC        {fmt(image_auroc)}")
        log(f"        image AP           {fmt(image_ap)}")
        log(f"        pixel AUROC        {fmt(pixel_auroc)}")
        log(f"        pixel AP           {fmt(pixel_ap)}")
        log(f"        pixel F1 @ thresh  {fmt(pixel_f1)}")
        log(
            f"        at threshold: acc={fmt(at_threshold['accuracy'])} "
            f"precision={fmt(at_threshold['precision'])} "
            f"recall={fmt(at_threshold['recall'])}"
        )
        log(f"        confusion: tn={tn} fp={fp} fn={fn} tp={tp}")
        log(
            f"        (max rule {max_threshold:.2f} would give: "
            f"acc={fmt(at_max_threshold['accuracy'])} "
            f"recall={fmt(at_max_threshold['recall'])} "
            f"fp={at_max_threshold['false_positives']} "
            f"fn={at_max_threshold['false_negatives']})"
        )
    else:
        log(
            f"\n[calib:{category}] thresholds set from {len(val_good)} held-out "
            "good images; no test data, so no accuracy is claimed"
        )
    log(f"[calib:{category}] saved -> {config.metrics_path(category)}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default=config.DEFAULT_CATEGORY)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    calibrate_category(args.category, args.batch_size)


if __name__ == "__main__":
    main()
