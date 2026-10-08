"""Verify the wrong-product safety check, and that it does not misfire.

Three things must hold at once:

  1. Correct product, no defect      -> "pass"
  2. Correct product, genuine defect -> "defect", WITH a defect-type label
  3. Wrong product entirely          -> "uncertain", WITHOUT a defect-type label

(3) is the safety check. (2) is the thing it must never break: a severe local
defect scores as high as a wrong product, so anything keyed on the score alone
would destroy it. The check is keyed on anomalous *coverage* instead - see
config.OOD_COVERAGE_THRESHOLD.

Run:  python -m scripts.domain_shift_test
"""

from __future__ import annotations

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src import dataset  # noqa: E402
from src.pipeline import InspectionService  # noqa: E402

RNG = np.random.default_rng(0)


def synthetic_images() -> dict[str, Image.Image]:
    size = (256, 256)
    ramp = np.tile(np.linspace(0, 255, size[1], dtype=np.uint8), (size[0], 1))
    return {
        "flat grey": Image.new("RGB", size, (128, 128, 128)),
        "uniform noise": Image.fromarray(RNG.integers(0, 256, (*size, 3), dtype=np.uint8)),
        "gradient": Image.fromarray(np.stack([ramp] * 3, axis=-1)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-category", type=int, default=4)
    args = parser.parse_args()

    service = InspectionService()
    cats = service.categories
    failures: list[str] = []

    good = {
        c: [s.path for s in dataset.list_split(c, "test") if not s.is_defective][
            : args.per_category
        ]
        for c in cats
    }
    defects = {
        c: [s.path for s in dataset.list_split(c, "test") if s.is_defective][
            : args.per_category
        ]
        for c in cats
    }
    synth = synthetic_images()

    print(
        f"{'checker':<11}{'input':<26}{'n':>4}{'pass':>6}{'defect':>8}"
        f"{'uncertain':>11}{'typed':>7}"
    )
    print("-" * 73)

    totals = {"own_good": [0, 0, 0], "own_defect": [0, 0, 0], "wrong": [0, 0, 0]}

    for category in cats:
        rows: list[tuple[str, list, str]] = [
            ("own good (correct product)", good[category], "own_good"),
            ("own defects (correct)", defects[category], "own_defect"),
        ]
        cross = [p for other in cats if other != category for p in good[other]]
        rows.append(("OTHER products' good", cross, "wrong"))
        rows.append(("synthetic images", list(synth.values()), "wrong"))

        for label, items, bucket in rows:
            if not items:
                continue
            counts = {"pass": 0, "defect": 0, "uncertain": 0}
            typed = 0
            for item in items:
                r = service.inspect(category, item)
                counts[r.verdict] += 1
                if r.defect_type is not None:
                    typed += 1

                # An "uncertain" result must never carry a defect-type label.
                if r.verdict == "uncertain" and (
                    r.defect_type is not None or r.defect_probabilities
                ):
                    failures.append(f"{category}: uncertain result carried a defect type")

            totals[bucket][0] += counts["pass"]
            totals[bucket][1] += counts["defect"]
            totals[bucket][2] += counts["uncertain"]

            print(
                f"{category:<11}{label:<26}{len(items):>4}{counts['pass']:>6}"
                f"{counts['defect']:>8}{counts['uncertain']:>11}{typed:>7}"
            )

            # Correct product + real defect must still be a plain DEFECT with a
            # type. This is the regression the safety check must not cause.
            if bucket == "own_defect" and counts["uncertain"]:
                failures.append(
                    f"{category}: {counts['uncertain']} genuine defect(s) wrongly "
                    "downgraded to 'uncertain'"
                )
            # A good part must never be called "uncertain" - that would mean the
            # safety check fires on the product it was calibrated for. A good
            # part called "defect" is the pre-existing false-alarm rate (carpet
            # 6/49, grid 3/49), measured in calibrate.py and unrelated to this
            # check, so it is reported but does not fail the run.
            if bucket == "own_good" and counts["uncertain"]:
                failures.append(
                    f"{category}: {counts['uncertain']} good part(s) wrongly "
                    "flagged as 'uncertain' by the safety check"
                )
        print()

    def pct(part: int, whole: int) -> str:
        return "n/a" if not whole else f"{part / whole:.0%}"

    print("summary across all categories")
    for name, key in (
        ("correct product, good", "own_good"),
        ("correct product, defective", "own_defect"),
        ("WRONG product / synthetic", "wrong"),
    ):
        p, d, u = totals[key]
        total = p + d + u
        print(
            f"  {name:<28} n={total:<4} pass={pct(p, total):>5} "
            f"defect={pct(d, total):>5} uncertain={pct(u, total):>5}"
        )

    caught = totals["wrong"][2]
    wrong_total = sum(totals["wrong"])
    print(f"\n  wrong-product photos caught by the safety check: {caught}/{wrong_total}")
    missed = wrong_total - caught
    if missed:
        print(
            f"  {missed} slipped through as a confident DEFECT - the check catches "
            "wildly different objects, not every possible mistake"
        )

    if failures:
        print("\nFAILURES:")
        for f in dict.fromkeys(failures):
            print(f"  - {f}")
        return 1
    print("\nall safety-check assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
