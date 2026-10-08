"""Run the full three-stage pipeline across several categories.

Run:  python -m src.run_all --categories capsule carpet grid metal_nut tile

Each category is independent, so a failure in one is reported and the rest
continue rather than losing the whole batch.

Uses one process for every category, which means the ImageNet backbone is
loaded once instead of six times. The memory bank is the only per-category
state, and it is written to disk before the next category starts.
"""

from __future__ import annotations

import argparse
import traceback
import warnings

warnings.filterwarnings("ignore")

from . import config  # noqa: E402
from .calibrate import calibrate_category  # noqa: E402
from .compare import build_comparison, print_table  # noqa: E402
from .train_classifier import train_category as train_classifier  # noqa: E402
from .train_patchcore import train_category as train_patchcore  # noqa: E402


def run_category(
    category: str,
    batch_size: int,
    accelerator: str,
    nested: bool,
    skip_trained: bool,
) -> None:
    banner = f"  {category}  "
    print(f"\n{'=' * 70}\n{banner:=^70}\n{'=' * 70}")

    if skip_trained and config.patchcore_ckpt(category).exists():
        print(f"[run:{category}] memory bank already exists, skipping PatchCore fit")
    else:
        train_patchcore(category, batch_size=batch_size, accelerator=accelerator)

    calibrate_category(category, batch_size=max(batch_size // 2, 1))
    train_classifier(category, nested=nested)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=config.CATEGORIES)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--nested-cv", action="store_true")
    parser.add_argument(
        "--skip-trained",
        action="store_true",
        help="Reuse an existing memory bank instead of refitting PatchCore.",
    )
    args = parser.parse_args()

    failures: dict[str, str] = {}
    for category in args.categories:
        try:
            run_category(
                category,
                args.batch_size,
                args.accelerator,
                args.nested_cv,
                args.skip_trained,
            )
        except Exception as exc:  # noqa: BLE001 - one bad category must not stop the batch
            failures[category] = f"{type(exc).__name__}: {exc}"
            print(f"\n[run:{category}] FAILED -> {failures[category]}")
            traceback.print_exc()

    print(f"\n{'=' * 70}\n{'  comparison  ':=^70}\n{'=' * 70}")
    rows = build_comparison()
    print_table(rows)

    if failures:
        print("\nFAILED CATEGORIES:")
        for category, message in failures.items():
            print(f"  {category}: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
