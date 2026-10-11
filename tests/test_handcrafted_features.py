import numpy as np

from src.features.handcrafted import FEATURE_NAMES, extract_features, segment_lesion


def _synthetic_lesion_image(size=(144, 192)):
    """A light background with a darker, slightly off-center blob, standing
    in for a dermoscopic image without needing a real file."""
    rng = np.random.default_rng(0)
    rgb = (rng.normal(200, 5, size=(*size, 3))).clip(0, 255).astype(np.uint8)

    yy, xx = np.mgrid[0 : size[0], 0 : size[1]]
    cy, cx = size[0] // 2 + 10, size[1] // 2 - 15
    blob = ((yy - cy) ** 2 / 30**2 + (xx - cx) ** 2 / 45**2) <= 1
    rgb[blob] = rng.normal(60, 5, size=(blob.sum(), 3)).clip(0, 255).astype(np.uint8)
    return rgb


def test_segment_lesion_finds_the_dark_blob():
    rgb = _synthetic_lesion_image()
    mask = segment_lesion(rgb)
    assert mask.dtype == bool
    assert 0 < mask.sum() < mask.size  # found something, not everything
    # the found region should be roughly centered where we drew the blob
    ys, xs = np.where(mask)
    assert abs(ys.mean() - (rgb.shape[0] // 2 + 10)) < 20
    assert abs(xs.mean() - (rgb.shape[1] // 2 - 15)) < 20


def test_extract_features_returns_all_named_features_with_no_nans():
    rgb = _synthetic_lesion_image()
    features = extract_features(rgb)
    assert list(features.keys()) == FEATURE_NAMES
    for name, value in features.items():
        assert np.isfinite(value), f"{name} is not finite: {value}"


def test_extract_features_handles_blank_image_without_crashing():
    flat = np.full((144, 192, 3), 128, dtype=np.uint8)
    features = extract_features(flat)
    assert list(features.keys()) == FEATURE_NAMES
    for name, value in features.items():
        assert np.isfinite(value), f"{name} is not finite: {value}"
