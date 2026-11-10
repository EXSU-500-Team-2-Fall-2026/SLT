"""Single-image prediction, tested with tiny stand-in models fitted on random
data and a synthetic image (no raw data or real weights needed)."""
import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

from src import predict
from src.eval.recalibration import save_calibrator
from src.features.handcrafted import FEATURE_NAMES
from src.models import classical_ml, metadata_baseline


@pytest.fixture()
def lesion_image(tmp_path):
    im = Image.new("RGB", (600, 450), (215, 170, 150))
    ImageDraw.Draw(im).ellipse((200, 130, 400, 320), fill=(90, 50, 35))
    path = tmp_path / "lesion.jpg"
    im.save(path)
    return path


@pytest.fixture()
def models_dir(tmp_path):
    rng = np.random.default_rng(0)
    out = tmp_path / "models"
    out.mkdir()

    y = rng.integers(0, 2, 120)
    features = pd.DataFrame(rng.normal(size=(120, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    joblib.dump(classical_ml.build_pipeline().fit(features, y), out / "classical_ml.joblib")

    meta = pd.DataFrame(
        {"age": rng.integers(20, 85, 120).astype(float), "sex": rng.choice(["male", "female"], 120),
         "localization": rng.choice(["back", "face", "trunk"], 120)}
    )
    joblib.dump(metadata_baseline.build_pipeline().fit(meta, y), out / "metadata_baseline.joblib")
    return out


def test_predicts_with_available_models_and_skips_missing_cnn(lesion_image, models_dir, tmp_path):
    loaded = predict.load_models(models_dir, tmp_path / "reports")
    assert set(loaded["models"]) == {"metadata_baseline", "classical_ml"}
    assert "cnn" in loaded["skipped"]

    results = predict.predict_image(lesion_image, loaded, age=62, sex="Male", localization="back")
    for name in ("metadata_baseline", "classical_ml"):
        assert 0.0 <= results[name]["raw"] <= 1.0
        assert results[name]["recalibrated"] is None
    assert "skipped" in predict.format_results(results, loaded["skipped"])


def test_missing_metadata_does_not_crash(lesion_image, models_dir, tmp_path):
    loaded = predict.load_models(models_dir, tmp_path / "reports")
    results = predict.predict_image(lesion_image, loaded)
    assert 0.0 <= results["metadata_baseline"]["raw"] <= 1.0


def test_unseen_body_site_does_not_crash(lesion_image, models_dir, tmp_path):
    loaded = predict.load_models(models_dir, tmp_path / "reports")
    results = predict.predict_image(lesion_image, loaded, age=40, sex="female", localization="elbow")
    assert 0.0 <= results["metadata_baseline"]["raw"] <= 1.0


def test_calibrator_is_applied_when_present(lesion_image, models_dir, tmp_path):
    save_calibrator({"intercept": -1.5, "slope": 1.0, "eps": 1e-6}, models_dir / "classical_ml_calibrator.json")
    loaded = predict.load_models(models_dir, tmp_path / "reports")
    r = predict.predict_image(lesion_image, loaded)["classical_ml"]
    assert r["recalibrated"] is not None
    assert r["recalibrated"] < r["raw"]


@pytest.mark.parametrize("content", [b"this is not an image", b""])
def test_invalid_file_gives_a_clear_error(tmp_path, models_dir, content):
    bad = tmp_path / "notes.jpg"
    bad.write_bytes(content)
    loaded = predict.load_models(models_dir, tmp_path / "reports")
    with pytest.raises(predict.InvalidImageError):
        predict.predict_image(bad, loaded)


def test_missing_file_gives_a_clear_error(tmp_path, models_dir):
    loaded = predict.load_models(models_dir, tmp_path / "reports")
    with pytest.raises(predict.InvalidImageError):
        predict.predict_image(tmp_path / "nope.jpg", loaded)


def _write_manifests(tmp_path):
    splits = tmp_path / "splits"
    splits.mkdir()
    pd.DataFrame([{"image_id": "ISIC_1", "lesion_id": "L1", "dx": "mel", "label": 1, "split": "val"}]).to_csv(
        splits / "val.csv", index=False)
    pd.DataFrame([{"image_id": "ISIC_9", "lesion_id": "T1", "dx": "nv", "label": 0, "split": "test"}]).to_csv(
        splits / "test.csv", index=False)
    return splits


def test_lookup_returns_truth_for_a_validation_image(tmp_path):
    info = predict.lookup_image_id("ISIC_1", tmp_path / "raw", [_write_manifests(tmp_path)])
    assert info["label"] == 1 and info["split"] == "val"
    assert info["path"].name == "ISIC_1.jpg"


def test_lookup_refuses_held_out_test_images(tmp_path):
    with pytest.raises(PermissionError, match="held-out test set"):
        predict.lookup_image_id("ISIC_9", tmp_path / "raw", [_write_manifests(tmp_path)])


def test_lookup_unknown_id(tmp_path):
    with pytest.raises(ValueError):
        predict.lookup_image_id("ISIC_404", tmp_path / "raw", [_write_manifests(tmp_path)])


def test_cnn_path_runs_end_to_end(lesion_image, tmp_path):
    torch = pytest.importorskip("torch")
    from src.models.cnn import SimpleCNN

    out = tmp_path / "models"
    out.mkdir()
    torch.save(SimpleCNN().state_dict(), out / "cnn.pt")
    loaded = predict.load_models(out, tmp_path / "reports")
    result = predict.predict_image(lesion_image, loaded)["cnn"]
    assert 0.0 <= result["raw"] <= 1.0
