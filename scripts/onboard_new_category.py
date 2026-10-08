"""Onboard a brand-new product category from a folder of photographs.

Developer-run, one command per client. Nothing here reimplements training or
calibration: it materialises the client's photos into the MVTec layout every
existing stage already understands, then calls the exact same functions the six
built-in categories were built with:

    train_patchcore.train_category  ->  calibrate.calibrate_category
                                    ->  train_classifier.train_category

Usage
-----
Detection only (no labelled defects yet):

    python -m scripts.onboard_new_category --category screws \
        --good-dir "C:/clients/acme/good_photos"

With labelled defect examples, one subfolder per defect type:

    python -m scripts.onboard_new_category --category screws \
        --good-dir "C:/clients/acme/good_photos" \
        --defects-dir "C:/clients/acme/defects"     # defects/<type>/*.png

The result lands in models/<category>/ and outputs/<category>/ exactly like the
built-in categories, so the API and the frontend category switcher pick it up on
next restart with no wiring.
"""

from __future__ import annotations

import argparse
import json
import shutil
import stat
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from PIL import Image  # noqa: E402

from src import config, dataset  # noqa: E402
from src.calibrate import calibrate_category  # noqa: E402
from src.compare import build_comparison  # noqa: E402
from src.train_classifier import train_category as train_classifier  # noqa: E402
from src.train_patchcore import train_category as train_patchcore  # noqa: E402

# PatchCore needs enough good images to build a representative memory bank, and
# the 10% calibration holdout needs enough left over to estimate a threshold.
MIN_GOOD_IMAGES = 50
# Share of good images reserved as test/good so a false-positive rate can be
# measured on parts the memory bank never saw.
TEST_GOOD_RATIO = 0.2


def find_images(folder: Path) -> list[Path]:
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in dataset.IMAGE_SUFFIXES
    )


def copy_all(paths: list[Path], destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        # copyfile, not copy2: copy2 carries the source's permission bits, and
        # read-only client media would then make the staged copy undeletable.
        shutil.copyfile(path, destination / path.name)
    return len(paths)


def _force_remove(path: Path) -> None:
    """rmtree that survives read-only files left by an earlier run."""

    def on_error(func, target, _exc):
        Path(target).chmod(stat.S_IWRITE)
        func(target)

    shutil.rmtree(path, onexc=on_error)


def write_blank_masks(image_paths: list[Path], destination: Path) -> None:
    """Create all-black ground-truth masks for defect images.

    anomalib's MVTec loader refuses to build a dataset whose defective test
    images have no matching mask, but a client supplying defect photographs will
    not have pixel-level annotations. Blank masks satisfy the loader; `calibrate`
    detects that they carry no positive pixels and reports pixel metrics as null
    rather than computing a meaningless number.
    """
    destination.mkdir(parents=True, exist_ok=True)
    for path in image_paths:
        with Image.open(path) as img:
            size = img.size
        Image.new("L", size, 0).save(destination / f"{path.stem}_mask.png")


def materialise(
    category: str, good_dir: Path, defects_dir: Path | None, force: bool
) -> dict:
    """Lay the client's photos out as <onboarded root>/<category>/{train,test}."""
    target = config.ONBOARDED_ROOT / category
    if target.exists():
        if not force:
            raise SystemExit(
                f"{target} already exists. Pass --force to replace it, or pick "
                "another --category name."
            )
        _force_remove(target)

    good = find_images(good_dir)
    if len(good) < MIN_GOOD_IMAGES:
        raise SystemExit(
            f"Found {len(good)} images in {good_dir}; need at least "
            f"{MIN_GOOD_IMAGES} defect-free photos to build a memory bank."
        )

    defect_sets: dict[str, list[Path]] = {}
    if defects_dir is not None:
        if not defects_dir.is_dir():
            raise SystemExit(f"--defects-dir not found: {defects_dir}")
        for sub in sorted(p for p in defects_dir.iterdir() if p.is_dir()):
            if sub.name == "good":
                continue  # good parts come from --good-dir
            images = find_images(sub)
            if images:
                defect_sets[sub.name] = images
        if not defect_sets:
            raise SystemExit(
                f"{defects_dir} has no defect-type subfolders containing images. "
                "Expected <defects-dir>/<defect_type>/*.png, or omit --defects-dir."
            )

    # Reserve a slice of good images for test/good. Deterministic (every Nth) so
    # a rerun produces the same split, and it samples across the whole folder
    # rather than taking one contiguous block.
    n_test = int(round(len(good) * TEST_GOOD_RATIO))
    step = max(len(good) // n_test, 1) if n_test else 0
    test_good = [good[i] for i in range(0, len(good), step)][:n_test] if n_test else []
    reserved = set(test_good)
    train_good = [p for p in good if p not in reserved]

    copy_all(train_good, target / "train" / "good")
    if test_good:
        copy_all(test_good, target / "test" / "good")
    for label, images in defect_sets.items():
        copy_all(images, target / "test" / label)
        write_blank_masks(images, target / "ground_truth" / label)

    return {
        "target": target,
        "n_good_total": len(good),
        "n_train_good": len(train_good),
        "n_test_good": len(test_good),
        "defect_classes": {k: len(v) for k, v in defect_sets.items()},
    }


# Below this many calibration images the MAD-based spread estimate gets
# unreliable; measured at 16 images it collapsed and the threshold came out too
# tight. Above ~20 it behaved.
MIN_CALIBRATION_IMAGES = 20
# Good photos below which the memory bank is too sparse to cover normal
# variation, so ordinary parts start scoring as novel.
RECOMMENDED_GOOD_IMAGES = 200
ACCEPTABLE_FPR = 0.05


def health_check(metrics: dict, n_good: int) -> list[str]:
    """Flag calibration that is statistically too thin to trust.

    The reserved test/good images were never in the memory bank, so the false
    alarm rate on them is a real, measurable signal - and the single most useful
    thing to show an operator before they deploy a freshly onboarded category.
    """
    issues: list[str] = []
    cal = metrics.get("calibration_scores", {})
    n_cal = cal.get("n", 0)

    if n_cal < MIN_CALIBRATION_IMAGES:
        issues.append(
            f"only {n_cal} calibration images (want >= {MIN_CALIBRATION_IMAGES}). "
            f"The threshold is estimated from their spread, and MAD is a noisy "
            f"estimator this small - it tends to come out too tight, causing "
            f"false alarms."
        )
    if n_good < RECOMMENDED_GOOD_IMAGES:
        issues.append(
            f"only {n_good} good photos (want >= {RECOMMENDED_GOOD_IMAGES}). "
            "A sparse memory bank does not cover normal part-to-part variation, "
            "so ordinary parts score as novel."
        )

    op = metrics.get("at_threshold") or {}
    seen_good = op.get("true_negatives", 0) + op.get("false_positives", 0)
    if seen_good:
        fpr = op["false_positives"] / seen_good
        if fpr > ACCEPTABLE_FPR:
            alt = metrics.get("at_max_threshold") or {}
            alt_fp = alt.get("false_positives")
            issues.append(
                f"false alarm rate {fpr:.0%} ({op['false_positives']}/{seen_good}) "
                f"on good images the bank never saw - above the {ACCEPTABLE_FPR:.0%} "
                f"target. The max-rule threshold would give {alt_fp}/{seen_good}. "
                "Collect more good photos before deploying this category."
            )
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--category", required=True, help="Name, e.g. 'screws'.")
    parser.add_argument(
        "--good-dir", required=True, type=Path, help="Folder of defect-free photos."
    )
    parser.add_argument(
        "--defects-dir",
        type=Path,
        default=None,
        help="Optional folder of labelled defects, one subfolder per defect type.",
    )
    parser.add_argument(
        "--type",
        choices=["object", "texture"],
        default="object",
        help="Rigid object or repeating texture. Affects grouping and reporting only.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing onboarded category."
    )
    args = parser.parse_args()

    category = args.category.strip().lower().replace(" ", "_")
    if category in config.BUILTIN_CATEGORIES:
        raise SystemExit(
            f"'{category}' is a built-in category. Choose a different name so the "
            "existing model is not overwritten."
        )
    if not args.good_dir.is_dir():
        raise SystemExit(f"--good-dir not found: {args.good_dir}")

    # A same-named folder in the MVTec root is shadowed by the staged copy.
    # Say so explicitly - silently training on the wrong images is exactly the
    # kind of thing that looks like it worked.
    shadowed = config.DATASET_ROOT / category
    if (shadowed / "train" / "good").is_dir():
        print(
            f"[onboard] note: {shadowed} also exists. The staged copy under "
            f"{config.ONBOARDED_ROOT} takes precedence, so training will use "
            "the photos you passed, not that folder."
        )

    started = time.perf_counter()
    print(f"\n=== Onboarding '{category}' ===")

    layout = materialise(category, args.good_dir, args.defects_dir, args.force)
    print(f"[onboard] staged -> {layout['target']}")
    print(
        f"[onboard] {layout['n_good_total']} good photos: "
        f"{layout['n_train_good']} for the memory bank "
        f"(10% of which is held out to calibrate), "
        f"{layout['n_test_good']} reserved as unseen good test images"
    )
    if layout["defect_classes"]:
        print(
            f"[onboard] {len(layout['defect_classes'])} defect types: "
            + ", ".join(f"{k} ({v})" for k, v in layout["defect_classes"].items())
        )
    else:
        print("[onboard] no labelled defects - detection and heatmaps only")

    # Record the type before calibration, which reads it back.
    config.outputs_dir(category).mkdir(parents=True, exist_ok=True)
    (config.outputs_dir(category) / "category.json").write_text(
        json.dumps(
            {
                "type": args.type,
                "onboarded": True,
                "source_good_dir": str(args.good_dir),
                "source_defects_dir": str(args.defects_dir) if args.defects_dir else None,
            },
            indent=2,
        )
    )

    # --- the same three stages the built-in categories use -------------------
    print("\n--- 1/3 PatchCore ---")
    train_patchcore(category, batch_size=args.batch_size, accelerator=args.accelerator)

    print("\n--- 2/3 Threshold calibration ---")
    metrics = calibrate_category(category, batch_size=max(args.batch_size // 2, 1))

    print("\n--- 3/3 Defect-type classifier ---")
    classifier = train_classifier(category, nested=True)

    elapsed = time.perf_counter() - started

    print(f"\n=== '{category}' onboarded in {elapsed / 60:.1f} min ===")
    print(f"  image threshold   {metrics['image_threshold']:.3f}")
    print(f"  pixel threshold   {metrics['pixel_threshold']:.3f}")
    if metrics.get("image_auroc") is not None:
        print(f"  image AUROC       {metrics['image_auroc']:.4f}")
        print(f"  detection acc.    {metrics['at_threshold']['accuracy']:.4f}")
    elif metrics.get("at_threshold"):
        op = metrics["at_threshold"]
        total = op["true_negatives"] + op["false_positives"]
        print(
            f"  false positives   {op['false_positives']}/{total} on unseen good "
            "images (no defects supplied, so recall is unknown)"
        )
    if classifier.get("trained"):
        acc = classifier.get("nested_cv_accuracy_mean") or classifier.get("cv_accuracy_mean")
        print(f"  defect types      {len(classifier['classes'])}, accuracy {acc:.3f}")
    else:
        print(f"  defect types      none - {classifier['cv_status']}")

    issues = health_check(metrics, layout["n_good_total"])
    if issues:
        print("\n  !! CALIBRATION WARNINGS -- review before deploying this category")
        for issue in issues:
            print(f"     - {issue}")

    # Refresh the cross-category table so the new entry shows up.
    rows = build_comparison(config.trained_categories())
    config.COMPARISON_PATH.write_text(json.dumps(rows, indent=2))
    print(f"\n  comparison table refreshed ({len(rows)} categories)")
    print("  restart the API to serve it: uvicorn src.api.main:app --port 8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
