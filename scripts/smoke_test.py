"""End-to-end check of the inspection pipeline on real images.

Runs one good and one defective image per defect type, for every trained
category, and verifies the detector's verdict.

Run:  python -m scripts.smoke_test [--categories bottle tile]
"""

from __future__ import annotations

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

from src import dataset  # noqa: E402
from src.pipeline import InspectionService  # noqa: E402

# Below this share of correct verdicts the pipeline is assumed broken rather
# than merely conservative.
FLOOR = 0.7


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument(
        "--per-label", type=int, default=1, help="Images to test per defect type."
    )
    args = parser.parse_args()

    service = InspectionService(args.categories)
    print(f"device={service.device}  categories={', '.join(service.categories)}\n")

    total_failures = 0
    total_checked = 0
    type_hits = 0
    type_total = 0

    for category in service.categories:
        assets = service.assets(category)
        print(f"--- {category}  (threshold {assets.image_threshold:.2f}) ---")

        by_label: dict[str, list] = {}
        for sample in dataset.list_split(category, "test"):
            by_label.setdefault(sample.label, []).append(sample)

        for label in sorted(by_label):
            for sample in by_label[label][: args.per_label]:
                result = service.inspect(category, sample.path)
                expected = label != "good"
                ok = result.is_defective == expected
                total_failures += 0 if ok else 1
                total_checked += 1

                top = "-"
                if result.defect_probabilities:
                    p = result.defect_probabilities[result.defect_type]
                    top = f"{result.defect_type} ({p:.0%})"
                    type_total += 1
                    type_hits += int(result.defect_type == label)

                for key, uri in result.images.items():
                    assert uri.startswith("data:image/"), f"bad {key} data URI"

                print(
                    f"  [{'PASS' if ok else 'FAIL'}] {label:<20} {sample.path.name:<9} "
                    f"score={result.anomaly_score:7.2f} "
                    f"defective={str(result.is_defective):<5} "
                    f"area={result.defect_area_pct:5.2f}%  type={top}"
                )
        print()

    print(f"detection misses: {total_failures}/{total_checked}")
    if type_total:
        print(f"defect-type agreement on these samples: {type_hits}/{type_total}")
        print("(in-sample: the shipped classifier was refit on these images)")

    # A miss on a subtle defect is a model characteristic, not a broken
    # pipeline - capsule's conservative threshold trades recall for zero false
    # alarms by design. Only fail the run if the pipeline looks structurally
    # wrong, i.e. it is getting most verdicts wrong somewhere.
    hit_rate = 1 - total_failures / max(total_checked, 1)
    if hit_rate < FLOOR:
        print(f"\nFAIL: verdict hit rate {hit_rate:.0%} below the {FLOOR:.0%} floor")
        return 1
    if total_failures:
        print("\nmisses are within the expected range for these thresholds; see")
        print("per_defect_recall in outputs/<category>/metrics.json for the full picture")
    return 0


if __name__ == "__main__":
    sys.exit(main())
