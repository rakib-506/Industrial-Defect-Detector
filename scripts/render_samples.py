"""Render an original / heatmap / outline montage per category.

Writes outputs/<category>/sample_results.png. Useful for eyeballing whether
localisation is landing on the real defect - especially for the texture
categories, where a "defect" is a disruption of a repeating pattern rather than
a change to a fixed object.

Run:  python -m scripts.render_samples [--categories carpet tile]
"""

from __future__ import annotations

import argparse
import base64
import io
import warnings

warnings.filterwarnings("ignore")

from PIL import Image, ImageDraw  # noqa: E402

from src import config, dataset  # noqa: E402
from src.pipeline import InspectionService  # noqa: E402

TILE = 240
PAD = 8
HEADER = 20


def render(service: InspectionService, category: str) -> None:
    by_label: dict[str, list] = {}
    for sample in dataset.list_split(category, "test"):
        by_label.setdefault(sample.label, []).append(sample)

    ordered = ["good"] + [label for label in sorted(by_label) if label != "good"]
    rows, captions = [], []
    for label in ordered:
        if label not in by_label:
            continue
        sample = by_label[label][0]
        result = service.inspect(category, sample.path)
        images = []
        for key in ("original", "heatmap", "outline"):
            raw = base64.b64decode(result.images[key].split(",", 1)[1])
            images.append(
                Image.open(io.BytesIO(raw)).convert("RGB").resize((TILE, TILE), Image.LANCZOS)
            )
        rows.append(images)
        captions.append(
            f"{label}   score={result.anomaly_score:.1f}   "
            f"{'DEFECT' if result.is_defective else 'PASS'}   "
            f"area={result.defect_area_pct:.1f}%   type={result.defect_type or '-'}"
        )

    width = 3 * TILE + 4 * PAD
    height = len(rows) * (TILE + HEADER + PAD) + PAD + HEADER
    canvas = Image.new("RGB", (width, height), (16, 20, 26))
    draw = ImageDraw.Draw(canvas)
    draw.text((PAD, 4), f"{category}   |   original / heatmap / outline", fill=(120, 200, 255))

    for index, (images, caption) in enumerate(zip(rows, captions)):
        y = PAD + HEADER + index * (TILE + HEADER + PAD)
        draw.text((PAD, y), caption, fill=(230, 237, 243))
        for column, image in enumerate(images):
            canvas.paste(image, (PAD + column * (TILE + PAD), y + HEADER))

    out = config.outputs_dir(category) / "sample_results.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"[render] {category}: {len(rows)} rows -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=None)
    args = parser.parse_args()

    service = InspectionService(args.categories)
    for category in service.categories:
        render(service, category)


if __name__ == "__main__":
    main()
