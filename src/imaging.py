"""Image loading, preprocessing and visualisation helpers.

Kept free of anomalib imports so the API layer stays light.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from . import config


def load_image(source: str | Path | bytes | Image.Image) -> Image.Image:
    """Load an arbitrary image source into an RGB PIL image."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source)).convert("RGB")
    return Image.open(source).convert("RGB")


_TRANSFORM = v2.Compose(
    [
        v2.Resize(list(config.IMAGE_SIZE), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        v2.Normalize(mean=list(config.IMAGENET_MEAN), std=list(config.IMAGENET_STD)),
    ]
)


def preprocess(image: Image.Image) -> torch.Tensor:
    """Resize to the model input size and normalise with ImageNet statistics.

    Deliberately mirrors anomalib's default PatchCore pre-processor
    (``Resize(256, bilinear, antialias) -> Normalize(imagenet)``, no centre
    crop) so serving-time inputs are identical to training-time ones.

    Returns a (1, 3, H, W) float tensor.
    """
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return _TRANSFORM(tensor).unsqueeze(0)


def _turbo_colormap(values: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to an RGB heatmap without pulling in matplotlib.

    Piecewise-linear approximation of the 'jet'/turbo family: blue -> cyan ->
    green -> yellow -> red.
    """
    anchors = np.array(
        [
            [0.0, 0.0, 0.20, 0.60],
            [0.25, 0.0, 0.70, 0.95],
            [0.50, 0.10, 0.85, 0.35],
            [0.75, 0.95, 0.85, 0.10],
            [1.0, 0.75, 0.05, 0.05],
        ]
    )
    values = np.clip(values, 0.0, 1.0)
    out = np.zeros((*values.shape, 3), dtype=np.float32)
    for channel in range(3):
        out[..., channel] = np.interp(values, anchors[:, 0], anchors[:, channel + 1])
    return out


def anomaly_map_to_overlay(
    image: Image.Image,
    anomaly_map: np.ndarray,
    vmin: float,
    vmax: float,
    alpha: float = 0.5,
) -> Image.Image:
    """Blend a normalised anomaly heatmap over the original image.

    ``vmin``/``vmax`` come from the calibrated score range so colours mean the
    same thing across different uploads rather than being per-image relative.
    """
    height, width = image.size[1], image.size[0]
    scaled = np.array(
        Image.fromarray(anomaly_map.astype(np.float32), mode="F").resize(
            (width, height), Image.BILINEAR
        )
    )
    denom = max(vmax - vmin, 1e-8)
    normalised = np.clip((scaled - vmin) / denom, 0.0, 1.0)

    heat = _turbo_colormap(normalised)
    base = np.asarray(image, dtype=np.float32) / 255.0
    # Fade the overlay in where the map is actually hot, so clean regions stay
    # readable instead of being tinted uniformly blue.
    weight = (normalised**1.5)[..., None] * alpha
    blended = base * (1 - weight) + heat * weight
    return Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))


def mask_to_outline(
    image: Image.Image,
    anomaly_map: np.ndarray,
    threshold: float,
    line_width: int = 3,
) -> Image.Image:
    """Draw the thresholded defect region as a contour over the image.

    The contour is thickened by ``line_width`` and the region inside it is given
    a light red wash, so the boundary stays readable on a full-resolution photo
    rather than disappearing as a hairline.
    """
    height, width = image.size[1], image.size[0]
    scaled = np.array(
        Image.fromarray(anomaly_map.astype(np.float32), mode="F").resize(
            (width, height), Image.BILINEAR
        )
    )
    mask = scaled >= threshold
    if not mask.any():
        return image.copy()

    # Morphological gradient: erode by `line_width`, then take the difference.
    eroded = mask
    for _ in range(max(line_width, 1)):
        padded = np.pad(eroded, 1, mode="edge")
        eroded = (
            padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:] & eroded
        )
    outline = mask & ~eroded

    out = np.asarray(image, dtype=np.float32).copy()
    accent = np.array([255.0, 60.0, 60.0], dtype=np.float32)
    out[mask] = out[mask] * 0.78 + accent * 0.22  # interior wash
    out[outline] = accent  # hard boundary
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def to_data_uri(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image as a base64 data URI for the JSON API."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"
