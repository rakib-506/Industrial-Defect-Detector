"""Re-run threshold calibration for every trained category.

Reuses each saved memory bank, so this is cheap compared with refitting
PatchCore. Existing classifier results in metrics.json are preserved.

Run:  python -m scripts.recalibrate_all
"""

from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

from src import config  # noqa: E402
from src.calibrate import calibrate_category  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Every trained category, including onboarded ones - not just the built-ins.
    parser.add_argument("--categories", nargs="+", default=config.trained_categories())
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    for category in args.categories:
        if not config.patchcore_ckpt(category).exists():
            print(f"[skip] {category}: no memory bank")
            continue
        calibrate_category(category, args.batch_size, quiet=True)
        print(f"[ok] {category}")


if __name__ == "__main__":
    main()
