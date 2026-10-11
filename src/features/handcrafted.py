"""Hand-crafted dermoscopic features for model 2/3 (classical ML).

No ground-truth lesion masks exist for the ISIC 2018 test set (only for
the HAM10000 train/val images), so we can't rely on them for features that
must be comparable across all three splits. Instead every image goes
through the same automatic pipeline: resize -> grayscale -> Otsu threshold
(lesions dermoscope darker than surrounding skin) -> largest connected
component. This is a simple approximation (~0.6 mean IoU against the real
HAM10000 masks), not pixel-accurate segmentation, but is good enough to
anchor shape/color/texture statistics to roughly the right region of the
image.

Features, loosely following the dermatology ABCD rule:
    - Asymmetry: 1 - IoU(mask, mask flipped), averaged over the horizontal
      and vertical axes.
    - Border: compactness (perimeter^2 / (4*pi*area)); 1.0 is a perfect
      circle, higher means a more irregular border.
    - Color: mean/std of R, G, B and H, S, V inside the lesion mask, plus
      the contrast between the lesion region and the surrounding skin.
    - Diameter/shape: area fraction of the image, eccentricity, solidity,
      extent.
    - Texture: GLCM contrast/homogeneity/energy/correlation on the
      grayscale crop around the lesion.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage import color as skcolor
from skimage import filters, measure, morphology
from skimage.feature import graycomatrix, graycoprops

RESIZE_SHAPE = (144, 192)  # (height, width)
GLCM_LEVELS = 16

FEATURE_NAMES = [
    "area_fraction",
    "eccentricity",
    "solidity",
    "extent",
    "compactness",
    "asymmetry",
    "mean_r",
    "mean_g",
    "mean_b",
    "std_r",
    "std_g",
    "std_b",
    "mean_h",
    "mean_s",
    "mean_v",
    "lesion_skin_contrast",
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
]


def load_and_resize(image_path) -> np.ndarray:
    im = Image.open(image_path).convert("RGB").resize(
        (RESIZE_SHAPE[1], RESIZE_SHAPE[0])
    )
    return np.asarray(im)


def segment_lesion(rgb: np.ndarray) -> np.ndarray:
    """Return a boolean mask of the largest dark blob, approximating the
    lesion. Falls back to an all-True mask if nothing is found."""
    gray = skcolor.rgb2gray(rgb)
    blurred = filters.gaussian(gray, sigma=2)
    thresh = filters.threshold_otsu(blurred)
    candidate = blurred < thresh

    candidate = morphology.remove_small_objects(candidate, max_size=199)
    candidate = morphology.closing(candidate, morphology.disk(3))

    labels = measure.label(candidate)
    if labels.max() == 0:
        return np.ones(gray.shape, dtype=bool)

    regions = measure.regionprops(labels)
    biggest = max(regions, key=lambda r: r.area)
    return labels == biggest.label


def _asymmetry_score(mask: np.ndarray) -> float:
    def iou_with(flipped: np.ndarray) -> float:
        inter = np.logical_and(mask, flipped).sum()
        union = np.logical_or(mask, flipped).sum()
        return inter / union if union else 1.0

    horizontal = iou_with(np.fliplr(mask))
    vertical = iou_with(np.flipud(mask))
    return 1.0 - (horizontal + vertical) / 2.0


def _color_stats(rgb: np.ndarray, mask: np.ndarray) -> dict:
    pixels = rgb[mask].astype(float)
    hsv = skcolor.rgb2hsv(rgb)[mask]
    outside = rgb[~mask].astype(float) if (~mask).any() else pixels

    return {
        "mean_r": pixels[:, 0].mean(),
        "mean_g": pixels[:, 1].mean(),
        "mean_b": pixels[:, 2].mean(),
        "std_r": pixels[:, 0].std(),
        "std_g": pixels[:, 1].std(),
        "std_b": pixels[:, 2].std(),
        "mean_h": hsv[:, 0].mean(),
        "mean_s": hsv[:, 1].mean(),
        "mean_v": hsv[:, 2].mean(),
        "lesion_skin_contrast": float(
            np.linalg.norm(pixels.mean(axis=0) - outside.mean(axis=0))
        ),
    }


def _shape_stats(mask: np.ndarray) -> dict:
    labels = measure.label(mask)
    if labels.max() == 0:
        return {
            "area_fraction": 0.0,
            "eccentricity": 0.0,
            "solidity": 1.0,
            "extent": 1.0,
            "compactness": 1.0,
        }
    region = max(measure.regionprops(labels), key=lambda r: r.area)
    area = region.area
    perimeter = max(region.perimeter, 1e-6)
    compactness = (perimeter**2) / (4 * np.pi * area)
    return {
        "area_fraction": area / mask.size,
        "eccentricity": region.eccentricity,
        "solidity": region.solidity,
        "extent": region.extent,
        "compactness": compactness,
    }


def _texture_stats(rgb: np.ndarray, mask: np.ndarray) -> dict:
    gray = skcolor.rgb2gray(rgb)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        crop = gray
    else:
        pad = 4
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, gray.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, gray.shape[1])
        crop = gray[y0:y1, x0:x1]

    quantized = (np.clip(crop, 0, 1) * (GLCM_LEVELS - 1)).astype(np.uint8)
    glcm = graycomatrix(
        quantized, distances=[1], angles=[0], levels=GLCM_LEVELS, symmetric=True, normed=True
    )
    return {
        "glcm_contrast": float(graycoprops(glcm, "contrast")[0, 0]),
        "glcm_homogeneity": float(graycoprops(glcm, "homogeneity")[0, 0]),
        "glcm_energy": float(graycoprops(glcm, "energy")[0, 0]),
        "glcm_correlation": float(np.nan_to_num(graycoprops(glcm, "correlation")[0, 0])),
    }


def extract_features(rgb: np.ndarray) -> dict:
    mask = segment_lesion(rgb)
    features = {"asymmetry": _asymmetry_score(mask)}
    features.update(_shape_stats(mask))
    features.update(_color_stats(rgb, mask))
    features.update(_texture_stats(rgb, mask))
    return {name: features[name] for name in FEATURE_NAMES}


def extract_features_from_path(image_path) -> dict:
    return extract_features(load_and_resize(image_path))
