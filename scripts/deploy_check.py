"""Simulate the deployed container without Docker.

Reproduces the three things that differ inside the image and are easy to get
wrong, because they only fail once deployed:

  1. The raw MVTec dataset and the staged client photos are NOT present -
     only models/, outputs/<cat>/metrics.json and deploy/samples/.
  2. There is no GPU, so the device must be CPU.
  3. The UI and the API are served by one app on one port, API under /api.

(1) and (2) are checked here directly. (3) needs a running server; start one with
`uvicorn src.api.main:app --port 7860` and run `scripts/api_test.py` against it.

Run:  python -m scripts.deploy_check
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Force the container's device before anything imports torch-backed modules.
os.environ.setdefault("DEFECT_DETECTOR_DEVICE", "cpu")

from src import config, dataset  # noqa: E402

# Pretend the raw dataset was never copied in - exactly what .dockerignore does.
config.DATASET_ROOTS = [config.SAMPLES_ROOT]

from src.pipeline import InspectionService  # noqa: E402


def main() -> int:
    failures: list[str] = []

    if not config.SAMPLES_ROOT.is_dir():
        print(f"FAIL: {config.SAMPLES_ROOT} missing - run `python -m scripts.export_samples`")
        return 1

    service = InspectionService()
    print(f"device   : {service.device}")
    print(f"categories: {len(service.categories)} -> {', '.join(service.categories)}")
    if str(service.device) != "cpu":
        failures.append(f"expected CPU in container, got {service.device}")
    if len(service.categories) != 7:
        failures.append(f"expected 7 categories, found {len(service.categories)}")

    print(f"\n{'category':<11}{'samples':>8}{'labels':>8}  {'verdicts on bundled samples'}")
    print("-" * 64)

    for category in service.categories:
        if not dataset.has_split(category, "test"):
            failures.append(f"{category}: no bundled samples reachable")
            print(f"{category:<11}{'0':>8}{'-':>8}  NO SAMPLES")
            continue

        samples = dataset.list_split(category, "test")
        labels = sorted({s.label for s in samples})

        verdicts: dict[str, int] = {}
        for sample in samples:
            result = service.inspect(category, sample.path)
            verdicts[result.verdict] = verdicts.get(result.verdict, 0) + 1

            if result.verdict == "uncertain":
                # A bundled sample is the correct product, so the wrong-product
                # check must not fire on it.
                failures.append(
                    f"{category}/{sample.label}/{sample.path.name}: bundled sample "
                    "flagged 'uncertain'"
                )
            for key, uri in result.images.items():
                if not uri.startswith("data:image/"):
                    failures.append(f"{category}: bad {key} image")

        summary = "  ".join(f"{k}={v}" for k, v in sorted(verdicts.items()))
        print(f"{category:<11}{len(samples):>8}{len(labels):>8}  {summary}")

    # The artefacts the image actually needs must all be present.
    print("\nrequired artefacts")
    for category in service.categories:
        missing = [
            str(p.relative_to(config.PROJECT_ROOT))
            for p in (config.patchcore_ckpt(category), config.metrics_path(category))
            if not p.exists()
        ]
        if missing:
            failures.append(f"{category}: missing {missing}")
    total_mb = sum(
        p.stat().st_size for p in config.MODELS_DIR.rglob("*") if p.is_file()
    ) / 1024 / 1024
    samples_mb = sum(
        p.stat().st_size for p in config.SAMPLES_ROOT.rglob("*") if p.is_file()
    ) / 1024 / 1024
    print(f"  models/        {total_mb:8.1f} MB")
    print(f"  deploy/samples {samples_mb:8.1f} MB")

    if failures:
        print("\nFAILURES:")
        for f in dict.fromkeys(failures):
            print(f"  - {f}")
        return 1
    print("\ncontainer simulation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
