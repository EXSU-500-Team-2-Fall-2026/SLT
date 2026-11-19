"""Grad-CAM: the example sampling and the overlay need no PyTorch; the heat
map tests run where PyTorch is installed."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.explain import gradcam


# ---------------------------------------------------------------- no PyTorch needed

def _predictions():
    return pd.DataFrame({
        "image_id": [f"ISIC_{i:03d}" for i in range(40)],
        "split": "tune",
        "label": [1] * 10 + [0] * 30,
        "prob": [0.9] * 6 + [0.2] * 4 + [0.8] * 7 + [0.1] * 23,
    })


def test_select_examples_covers_each_outcome_group():
    chosen = gradcam.select_examples(_predictions(), threshold=0.5, per_group=2)
    assert chosen["outcome"].value_counts().to_dict() == {group: 2 for group in gradcam.OUTCOME_GROUPS}
    by_outcome = {group: chosen[chosen["outcome"] == group] for group in gradcam.OUTCOME_GROUPS}
    assert (by_outcome["true positive"]["label"] == 1).all() and (by_outcome["true positive"]["prob"] >= 0.5).all()
    assert (by_outcome["false negative"]["label"] == 1).all() and (by_outcome["false negative"]["prob"] < 0.5).all()
    assert (by_outcome["false positive"]["label"] == 0).all() and (by_outcome["false positive"]["prob"] >= 0.5).all()
    assert (by_outcome["true negative"]["label"] == 0).all() and (by_outcome["true negative"]["prob"] < 0.5).all()


def test_select_examples_is_reproducible_and_seed_dependent():
    first = gradcam.select_examples(_predictions(), 0.5, per_group=2, seed=42)
    again = gradcam.select_examples(_predictions(), 0.5, per_group=2, seed=42)
    other = gradcam.select_examples(_predictions(), 0.5, per_group=2, seed=7)
    assert list(first["image_id"]) == list(again["image_id"])
    assert list(first["image_id"]) != list(other["image_id"])


def test_select_examples_handles_small_groups():
    predictions = _predictions()
    predictions = predictions[~((predictions["label"] == 1) & (predictions["prob"] < 0.5))]  # no false negatives
    chosen = gradcam.select_examples(predictions, 0.5, per_group=2)
    assert "false negative" not in set(chosen["outcome"]) and len(chosen) == 6


def test_select_examples_refuses_test_predictions():
    predictions = _predictions()
    predictions.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="held-out test"):
        gradcam.select_examples(predictions, 0.5)


def test_overlay_keeps_size_and_leaves_zero_regions_untouched():
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, size=(45, 60, 3), dtype=np.uint8))
    cam = np.zeros((96, 96), dtype=np.float32)
    cam[:48, :48] = 1.0  # top-left quarter hot

    overlay = gradcam.overlay_heatmap(image, cam)
    assert overlay.size == image.size and overlay.mode == "RGB"
    original, blended = np.asarray(image).astype(int), np.asarray(overlay).astype(int)
    assert np.array_equal(blended[30:, 40:], original[30:, 40:])        # cold corner unchanged
    assert not np.array_equal(blended[:15, :20], original[:15, :20])    # hot corner tinted

    untouched = gradcam.overlay_heatmap(image, np.zeros((96, 96), dtype=np.float32))
    assert np.array_equal(np.asarray(untouched), np.asarray(image))


# ---------------------------------------------------------------- PyTorch needed

def _identity_cnn(torch):
    """SimpleCNN rewired so that its score is the average brightness of the
    red channel after the four pooling steps: where the red channel is
    bright is then exactly where the score comes from."""
    from src.models.cnn import SimpleCNN

    model = SimpleCNN().eval()
    with torch.no_grad():
        for block in model.features:
            convolution, norm = block[0], block[1]
            convolution.weight.zero_()
            convolution.bias.zero_()
            convolution.weight[0, 0, 1, 1] = 1.0  # pass channel 0 through unchanged
            norm.reset_parameters()
            norm.reset_running_stats()
        first, second = model.head[2], model.head[5]
        for linear in (first, second):
            linear.weight.zero_()
            linear.bias.zero_()
            linear.weight[0, 0] = 1.0
    return model


def test_heat_map_points_at_the_region_that_drives_the_score():
    torch = pytest.importorskip("torch")
    model = _identity_cnn(torch)
    image = torch.zeros(3, 96, 96)
    image[0, 8:40, 56:88] = 1.0  # bright red square in the top-right quarter

    cam, probability = gradcam.grad_cam(model, image)
    assert cam.shape == (96, 96) and cam.dtype == np.float32
    assert cam.max() == pytest.approx(1.0) and cam.min() >= 0.0
    row, column = np.unravel_index(cam.argmax(), cam.shape)
    assert row < 48 and column >= 48
    assert cam[:48, 48:].mean() > 10 * cam[48:, :48].mean() + 1e-6
    assert 0.5 < probability < 1.0


def test_probability_matches_a_normal_forward_pass_and_model_is_left_unchanged():
    torch = pytest.importorskip("torch")
    from src.models.cnn import SimpleCNN

    torch.manual_seed(0)
    model = SimpleCNN()
    model.train()  # deliberately left in training mode by the caller
    image = torch.randn(3, 96, 96)

    cam, probability = gradcam.grad_cam(model, image)
    assert model.training  # restored
    assert all(parameter.grad is None for parameter in model.parameters())

    model.eval()
    with torch.no_grad():
        expected = float(torch.sigmoid(model(image.unsqueeze(0)))[0])
    assert probability == pytest.approx(expected, abs=1e-6)
    assert cam.shape == (96, 96) and np.isfinite(cam).all() and 0.0 <= cam.min() and cam.max() <= 1.0

    again, _ = gradcam.grad_cam(model, image)
    assert np.allclose(cam, again)


def test_works_with_frozen_weights_and_other_input_sizes():
    torch = pytest.importorskip("torch")
    from src.models.cnn import SimpleCNN

    torch.manual_seed(1)
    model = SimpleCNN().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cam, _ = gradcam.grad_cam(model, torch.randn(1, 3, 64, 80))
    assert cam.shape == (64, 80)


def test_empty_map_when_nothing_raises_the_score():
    torch = pytest.importorskip("torch")
    model = _identity_cnn(torch)
    with torch.no_grad():
        model.head[5].weight[0, 0] = -1.0  # brightness now lowers the score
    image = torch.zeros(3, 96, 96)
    image[0, 8:40, 56:88] = 1.0
    cam, probability = gradcam.grad_cam(model, image)
    assert cam.max() == 0.0 and probability < 0.5


def test_explain_image_end_to_end(tmp_path):
    torch = pytest.importorskip("torch")
    from src.models.cnn import SimpleCNN

    torch.manual_seed(2)
    rng = np.random.default_rng(2)
    path = tmp_path / "lesion.jpg"
    Image.fromarray(rng.integers(0, 255, size=(450, 600, 3), dtype=np.uint8)).save(path)

    image, overlay, probability, cam = gradcam.explain_image(path, SimpleCNN().eval(), image_size=96)
    assert image.size == (600, 450) and overlay.size == (600, 450)
    assert cam.shape == (96, 96) and 0.0 <= probability <= 1.0

    gradcam.save_side_by_side(image, overlay, "test", tmp_path / "out" / "single.png")
    gradcam.save_panel(
        [{"image": image, "overlay": overlay, "outcome": "true positive", "dx": "mel", "probability": probability}] * 3,
        tmp_path / "out" / "panel.png",
    )
    assert (tmp_path / "out" / "single.png").stat().st_size > 0 and (tmp_path / "out" / "panel.png").stat().st_size > 0
