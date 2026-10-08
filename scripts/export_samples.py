"""Export a small demo gallery that can ship inside a container image.

The deployed image does not carry the raw MVTec folders - they are far too large
and are excluded by .dockerignore. Without them the sample gallery would be empty,
so this writes a few downscaled JPEGs per category into `deploy/samples/` in the
same `<category>/test/<label>/` layout the dataset loader already understands.
`config.SAMPLES_ROOT` is the last dataset root, so these are used only when the
real dataset is absent.

Run:  python -m scripts.export_samples
"""

from __future__ import annotations

import argparse
import shutil
import warnings

warnings.filterwarnings("ignore")

from PIL import Image  # noqa: E402

from src import config, dataset  # noqa: E402

# Big enough for the gallery thumbnail and for a real inspection, small enough
# that 7 categories stay well under ~25 MB.
MAX_EDGE = 700
JPEG_QUALITY = 82


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-label", type=int, default=2, help="Images per class, per category."
    )
    parser.add_argument("--clean", action="store_true", help="Remove existing samples first.")
    args = parser.parse_args()

    if args.clean and config.SAMPLES_ROOT.exists():
        shutil.rmtree(config.SAMPLES_ROOT)

    total = 0
    total_bytes = 0
    for category in config.trained_categories():
        # Read through the normal roots, but skip SAMPLES_ROOT itself so a rerun
        # does not re-encode its own output.
        try:
            samples = dataset.list_split(category, "test")
        except FileNotFoundError:
            print(f"[export] {category}: no test split available, skipped")
            continue
        if not samples:
            continue

        source_root = config.category_dir(category)
        if source_root.is_relative_to(config.SAMPLES_ROOT):
            print(f"[export] {category}: only bundled samples on disk, left as-is")
            continue

        per_label: dict[str, list] = {}
        for sample in samples:
            per_label.setdefault(sample.label, []).append(sample)

        count = 0
        for label in sorted(per_label):
            for sample in per_label[label][: args.per_label]:
                out_dir = config.SAMPLES_ROOT / category / "test" / label
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{sample.path.stem}.jpg"

                with Image.open(sample.path) as img:
                    img = img.convert("RGB")
                    img.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
                    img.save(out_path, "JPEG", quality=JPEG_QUALITY, optimize=True)

                total_bytes += out_path.stat().st_size
                count += 1
        total += count
        print(f"[export] {category}: {count} images across {len(per_label)} classes")

    print(f"\n[export] {total} images, {total_bytes / 1024 / 1024:.1f} MB -> {config.SAMPLES_ROOT}")


if __name__ == "__main__":
    main()
