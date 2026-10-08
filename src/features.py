"""Feature extraction for the defect-type classifier.

The classifier answers a different question from PatchCore: given that a part is
anomalous, *what kind* of defect is it? It therefore looks only at the region
PatchCore flagged, described two ways:

1. Appearance - the PatchCore patch embeddings, pooled with the anomaly map as
   attention weights, so the vector describes the defect rather than the bottle.
2. Geometry - shape statistics of the thresholded region. These carry most of
   the signal between the MVTec bottle classes: `broken_large` is one big
   contiguous area, `broken_small` a small one, `contamination` tends to be
   small, rounder and often multi-blob.

Both are shared verbatim between training and serving so there is no skew.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

GEOMETRIC_NAMES = [
    "area_fraction",
    "max_score",
    "region_mean_score",
    "region_score_std",
    "num_components",
    "largest_component_fraction",
    "compactness",
    "bbox_aspect",
    "bbox_fill",
    "eccentricity",
    "centroid_radius",
    "map_p99",
    "map_mean",
]


def attention_pool(embedding: torch.Tensor, anomaly_map: np.ndarray, sharpness: float = 4.0) -> np.ndarray:
    """Pool a (C, h, w) patch-embedding grid using the anomaly map as weights.

    ``sharpness`` raises the normalised weights to a power so pooling
    concentrates on the hottest patches instead of averaging the whole bottle.
    """
    if embedding.dim() == 4:
        embedding = embedding[0]
    channels, height, width = embedding.shape

    weights = torch.from_numpy(np.ascontiguousarray(anomaly_map)).float()[None, None]
    weights = F.interpolate(weights, size=(height, width), mode="bilinear", align_corners=False)[0, 0]

    lo, hi = float(weights.min()), float(weights.max())
    weights = (weights - lo) / max(hi - lo, 1e-8)
    weights = weights.clamp(min=0) ** sharpness
    total = float(weights.sum())
    if total <= 1e-8:  # completely flat map - fall back to uniform pooling
        weights = torch.ones_like(weights)
        total = float(weights.sum())

    pooled = (embedding * weights[None]).sum(dim=(1, 2)) / total
    pooled = pooled / max(float(pooled.norm()), 1e-8)  # L2 normalise
    return pooled.numpy().astype(np.float32)


def geometric_features(anomaly_map: np.ndarray, pixel_threshold: float) -> np.ndarray:
    """Shape and intensity statistics of the thresholded anomalous region."""
    amap = np.asarray(anomaly_map, dtype=np.float32)
    total_pixels = amap.size
    mask = amap >= pixel_threshold

    max_score = float(amap.max())
    map_p99 = float(np.percentile(amap, 99))
    map_mean = float(amap.mean())

    if not mask.any():
        # Nothing crossed the threshold; report a well-defined empty region so
        # the feature vector keeps a stable length and scale.
        return np.array(
            [0.0, max_score, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, map_p99, map_mean],
            dtype=np.float32,
        )

    area_fraction = float(mask.sum()) / total_pixels
    region_values = amap[mask]
    region_mean = float(region_values.mean())
    region_std = float(region_values.std())

    labels, num_components = ndimage.label(mask)
    component_sizes = np.bincount(labels.ravel())[1:]
    largest = int(component_sizes.argmax()) + 1
    largest_fraction = float(component_sizes.max()) / total_pixels

    # Describe the dominant blob only; secondary specks are captured by the
    # component count instead.
    blob = labels == largest
    rows, cols = np.nonzero(blob)
    height = rows.max() - rows.min() + 1
    width = cols.max() - cols.min() + 1
    blob_area = float(blob.sum())
    bbox_aspect = float(max(height, width) / max(min(height, width), 1))
    bbox_fill = blob_area / float(height * width)

    perimeter = float(np.abs(np.diff(blob.astype(np.int8), axis=0)).sum() + np.abs(np.diff(blob.astype(np.int8), axis=1)).sum())
    # 1.0 for a perfect circle, lower for ragged or elongated shapes.
    compactness = float(4 * np.pi * blob_area / max(perimeter**2, 1e-8))

    # Eccentricity from the second central moments of the dominant blob.
    row_c, col_c = rows.mean(), cols.mean()
    dr, dc = rows - row_c, cols - col_c
    cov = np.array([[(dr * dr).mean(), (dr * dc).mean()], [(dr * dc).mean(), (dc * dc).mean()]])
    eigenvalues = np.clip(np.linalg.eigvalsh(cov), 0, None)
    major, minor = float(eigenvalues.max()), float(eigenvalues.min())
    eccentricity = float(np.sqrt(1 - minor / major)) if major > 1e-8 else 0.0

    # How far off-centre the defect sits, normalised by the half-diagonal.
    centre = np.array(amap.shape, dtype=np.float32) / 2
    centroid_radius = float(
        np.linalg.norm(np.array([row_c, col_c]) - centre) / np.linalg.norm(centre)
    )

    return np.array(
        [
            area_fraction,
            max_score,
            region_mean,
            region_std,
            float(num_components),
            largest_fraction,
            compactness,
            bbox_aspect,
            bbox_fill,
            eccentricity,
            centroid_radius,
            map_p99,
            map_mean,
        ],
        dtype=np.float32,
    )


def build_feature_vector(
    embedding: torch.Tensor, anomaly_map: np.ndarray, pixel_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(appearance, geometry)`` feature blocks for one image."""
    return (
        attention_pool(embedding, anomaly_map),
        geometric_features(anomaly_map, pixel_threshold),
    )
