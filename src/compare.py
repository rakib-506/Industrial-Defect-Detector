"""Build the cross-category comparison table.

Reads each category's `outputs/<category>/metrics.json` and emits a single
`outputs/comparison.json` plus a printable table.

Run:  python -m src.compare
"""

from __future__ import annotations

import argparse
import json

from . import config, dataset


def _fmt(value: float | None, spec: str = ".4f", missing: str = "n/a") -> str:
    return missing if value is None else format(value, spec)


def build_comparison(categories: list[str] | None = None) -> list[dict]:
    """Collect one row per category that has been evaluated."""
    categories = categories or config.trained_categories()
    rows: list[dict] = []
    for category in categories:
        path = config.metrics_path(category)
        if not path.exists():
            continue
        m = json.loads(path.read_text())
        clf = m.get("classifier") or {}

        # Prefer the unbiased nested estimate; fall back to plain CV and say so.
        accuracy = clf.get("nested_cv_accuracy_mean")
        accuracy_std = clf.get("nested_cv_accuracy_std")
        accuracy_kind = "nested-CV"
        if accuracy is None:
            accuracy = clf.get("cv_accuracy_mean")
            accuracy_std = clf.get("cv_accuracy_std")
            accuracy_kind = "CV" if accuracy is not None else "none"

        rows.append(
            {
                "category": category,
                "type": config.category_type(category),
                "onboarded": config.is_onboarded(category),
                "has_classifier": bool(clf.get("trained", bool(clf.get("classes")))),
                "has_defect_labels": m.get("has_defect_labels", True),
                "image_auroc": m.get("image_auroc"),
                "pixel_auroc": m.get("pixel_auroc"),
                "pixel_f1": m.get("pixel_f1_at_threshold"),
                "detection_accuracy": (m.get("at_threshold") or {}).get("accuracy"),
                "detection_recall": (m.get("at_threshold") or {}).get("recall"),
                "false_positives": (m.get("at_threshold") or {}).get("false_positives"),
                "false_negatives": (m.get("at_threshold") or {}).get("false_negatives"),
                "maxrule_accuracy": (m.get("at_max_threshold") or {}).get("accuracy"),
                "maxrule_recall": (m.get("at_max_threshold") or {}).get("recall"),
                "maxrule_false_positives": (m.get("at_max_threshold") or {}).get(
                    "false_positives"
                ),
                "threshold_rule": m.get("threshold_rule"),
                "classifier_accuracy": accuracy,
                "classifier_accuracy_std": accuracy_std,
                "classifier_accuracy_kind": accuracy_kind,
                "defect_classes": len(clf.get("classes") or dataset.defect_labels(category)),
                "min_per_class": clf.get("min_per_class"),
                "n_defect_images": clf.get("n_samples"),
                "cv_status": clf.get("cv_status"),
                "nested_cv_status": clf.get("nested_cv_status"),
                "layers": (m.get("patchcore") or {}).get("layers"),
                "memory_bank": (m.get("patchcore") or {}).get("memory_bank"),
            }
        )
    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No evaluated categories found. Run the pipeline first.")
        return

    width = max(12, max(len(r["category"]) for r in rows) + 2)
    header = (
        f"{'category':<{width}}{'type':<9}{'img AUROC':>11}{'px AUROC':>10}"
        f"{'px F1':>8}{'det acc':>9}{'classes':>9}{'clf acc':>18}"
    )
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: (r["type"], r["category"])):
        if row["classifier_accuracy"] is None:
            clf_cell = "no labels" if not row.get("has_defect_labels", True) else "insufficient"
        else:
            std = row["classifier_accuracy_std"] or 0.0
            tag = "" if row["classifier_accuracy_kind"] == "nested-CV" else "*"
            clf_cell = f"{row['classifier_accuracy']:.3f} +/- {std:.3f}{tag}"
        print(
            f"{row['category']:<{width}}{row['type']:<9}"
            f"{_fmt(row['image_auroc'], '.4f'):>11}"
            f"{_fmt(row['pixel_auroc'], '.4f'):>10}"
            f"{_fmt(row['pixel_f1'], '.3f'):>8}"
            f"{_fmt(row['detection_accuracy'], '.3f'):>9}"
            f"{row['defect_classes']:>9}"
            f"{clf_cell:>18}"
        )

    if any(r["classifier_accuracy_kind"] == "CV" for r in rows):
        print("\n* plain cross-validation (nested CV unavailable); mildly optimistic.")

    # Show what the simpler max-over-held-out-good rule would have given, so the
    # choice of threshold estimator is visible rather than buried.
    alts = [
        r
        for r in rows
        if r.get("maxrule_accuracy") is not None and r.get("detection_accuracy") is not None
    ]
    if alts:
        print("\nthreshold rule (both calibrated on held-out good images only):")
        print(
            f"  {'category':<{width}}{'robust acc':>12}{'robust fp':>11}"
            f"{'max-rule acc':>14}{'max-rule fp':>13}"
        )
        for r in sorted(alts, key=lambda x: x["category"]):
            print(
                f"  {r['category']:<{width}}{r['detection_accuracy']:>12.3f}"
                f"{r['false_positives']:>11}"
                f"{r['maxrule_accuracy']:>14.3f}{r['maxrule_false_positives']:>13}"
            )
        builtin = [r for r in alts if not r.get("onboarded")]
        if builtin:
            n = len(builtin)
            print(
                f"  {'mean (built-in)':<{width}}"
                f"{sum(r['detection_accuracy'] for r in builtin) / n:>12.3f}"
                f"{'':>11}{sum(r['maxrule_accuracy'] for r in builtin) / n:>14.3f}"
            )

    # The object-vs-texture comparison is about the six MVTec categories, which
    # all ran identical settings on identically prepared data. Onboarded
    # categories use a different good/test split and may lack masks entirely, so
    # folding them in would silently change what the aggregate means.
    def mean_of(kind: str, field: str) -> float | None:
        values = [
            r[field]
            for r in rows
            if r["type"] == kind and not r.get("onboarded") and r[field] is not None
        ]
        return sum(values) / len(values) if values else None

    parts = []
    for field, label in (("image_auroc", "image"), ("pixel_auroc", "pixel")):
        obj, tex = mean_of("object", field), mean_of("texture", field)
        if obj is not None and tex is not None:
            parts.append(f"mean {label} AUROC  objects={obj:.4f}  textures={tex:.4f}")
    if parts:
        print("\n" + "\n".join(parts) + "\n  (built-in categories only)")

    onboarded = [r for r in rows if r.get("onboarded")]
    if onboarded:
        print(f"\nonboarded categories ({len(onboarded)}):")
        for r in sorted(onboarded, key=lambda x: x["category"]):
            bits = []
            if r["image_auroc"] is None:
                bits.append("no labelled defects, so detection accuracy is unmeasured")
            if r["pixel_auroc"] is None and r["image_auroc"] is not None:
                bits.append("no ground-truth masks, so pixel metrics are unavailable")
            print(f"  {r['category']:<17}{'; '.join(bits) or 'fully evaluated'}")

    for row in rows:
        for key in ("cv_status", "nested_cv_status"):
            status = row.get(key)
            if status and status not in ("ok", "not run"):
                print(f"\n[{row['category']}] {key}: {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=None)
    args = parser.parse_args()

    rows = build_comparison(args.categories)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.COMPARISON_PATH.write_text(json.dumps(rows, indent=2))
    print_table(rows)
    print(f"\nsaved -> {config.COMPARISON_PATH}")


if __name__ == "__main__":
    main()
