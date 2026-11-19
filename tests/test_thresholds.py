"""Freezing decision thresholds: the rule itself, the guards, the files
written, and how src/predict.py uses them."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

from src import predict
from src.eval import freeze_thresholds as ft
from src.eval.recalibration import apply_recalibration, save_calibrator
from src.features.handcrafted import FEATURE_NAMES
from src.models import classical_ml


# ---------------------------------------------------------------- the rule

def test_threshold_is_the_score_of_the_right_positive_case():
    y = np.array([1] * 10 + [0] * 10)
    scores = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.10] + [0.5] * 10)
    assert ft.threshold_for_sensitivity(y, scores, 0.90) == 0.55   # 9 of 10 positives at or above
    assert ft.threshold_for_sensitivity(y, scores, 0.80) == 0.60
    assert ft.threshold_for_sensitivity(y, scores, 1.00) == 0.10
    assert ft.threshold_for_sensitivity(y, scores, 0.85) == 0.55   # 8.5 rounds up to 9 positives


def test_threshold_meets_the_target_and_is_the_highest_that_does():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    scores = np.clip(0.25 * y + rng.uniform(0, 0.75, 500), 0, 1)
    for target in (0.5, 0.9, 0.95):
        threshold = ft.threshold_for_sensitivity(y, scores, target)
        assert ft.operating_point(y, scores, threshold)["sensitivity"] >= target
        higher = np.nextafter(threshold, 1.0)
        assert ft.operating_point(y, scores, higher)["sensitivity"] < target


def test_ties_at_the_threshold_are_all_flagged():
    y = np.array([1, 1, 1, 1, 0, 0])
    scores = np.array([0.9, 0.4, 0.4, 0.4, 0.4, 0.1])
    threshold = ft.threshold_for_sensitivity(y, scores, 0.5)
    assert threshold == 0.4
    point = ft.operating_point(y, scores, threshold)
    assert point["sensitivity"] == 1.0 and point["specificity"] == 0.5 and point["n_positive"] == 4


@pytest.mark.parametrize("target", [0, -0.1, 1.5])
def test_invalid_targets_are_rejected(target):
    with pytest.raises(ValueError):
        ft.threshold_for_sensitivity([1, 0], [0.9, 0.1], target)


def test_no_positive_cases_is_an_error():
    with pytest.raises(ValueError, match="no positive"):
        ft.threshold_for_sensitivity([0, 0], [0.9, 0.1], 0.9)


# ---------------------------------------------------------------- freezing from saved files

@pytest.fixture()
def reports(tmp_path):
    rng = np.random.default_rng(1)
    for model in ("metadata_baseline", "classical_ml"):
        folder = tmp_path / "reports" / model
        folder.mkdir(parents=True)
        for split in ("tune", "calibration"):
            y = rng.integers(0, 2, 200)
            frame = pd.DataFrame({"image_id": [f"{split}_{i}" for i in range(200)], "lesion_id": "L", "group_id": "G",
                                  "split": split, "label": y,
                                  "prob": np.clip(0.3 * y + rng.uniform(0, 0.7, 200), 0.001, 0.999)})
            frame.to_csv(folder / f"predictions_{split}.csv", index=False)
        save_calibrator({"intercept": -1.3, "slope": 0.9, "eps": 1e-6}, folder / "calibrator.json")
    return tmp_path / "reports"


def test_frozen_threshold_contents(reports):
    frozen = ft.freeze_model_threshold("classical_ml", reports, target=0.90)
    tune = pd.read_csv(reports / "classical_ml" / "predictions_tune.csv")

    assert frozen["selected_on"] == "tune" and frozen["target_sensitivity"] == 0.90
    assert frozen["tune"]["sensitivity"] >= 0.90 and frozen["tune"]["n"] == 200
    assert frozen["threshold_raw"] in set(tune.loc[tune["label"] == 1, "prob"])
    assert frozen["threshold_recalibrated"] < frozen["threshold_raw"]  # this calibrator lowers probabilities

    # both scales flag exactly the same cases
    calibrator = json.load(open(reports / "classical_ml" / "calibrator.json"))
    by_raw = tune["prob"].to_numpy() >= frozen["threshold_raw"]
    by_recalibrated = apply_recalibration(tune["prob"].to_numpy(), calibrator) >= frozen["threshold_recalibrated"] - 1e-12
    assert np.array_equal(by_raw, by_recalibrated)


@pytest.mark.parametrize("split, message", [("test", "held-out test set"), ("calibration", "reserved for fitting")])
def test_refuses_test_and_calibration_subsets(reports, split, message):
    with pytest.raises(ValueError, match=message):
        ft.freeze_model_threshold("classical_ml", reports, split=split)


def test_refuses_predictions_from_mixed_subsets(reports):
    path = reports / "classical_ml" / "predictions_tune.csv"
    mixed = pd.read_csv(path)
    mixed.loc[0, "split"] = "train"
    mixed.to_csv(path, index=False)
    with pytest.raises(ValueError, match="other splits"):
        ft.freeze_model_threshold("classical_ml", reports)


def test_refuses_a_calibrator_that_reverses_the_ranking(reports):
    save_calibrator({"intercept": 0.0, "slope": -0.5, "eps": 1e-6}, reports / "classical_ml" / "calibrator.json")
    with pytest.raises(ValueError, match="slope is not positive"):
        ft.freeze_model_threshold("classical_ml", reports)


def test_needs_a_fitted_calibrator(reports):
    (reports / "classical_ml" / "calibrator.json").rename(reports / "classical_ml" / "elsewhere.json")
    with pytest.raises(FileNotFoundError):
        ft.freeze_model_threshold("classical_ml", reports)


def test_main_writes_both_copies_and_skips_missing_models(reports, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["freeze", "--reports-dir", str(reports), "--models-dir", str(tmp_path / "models")])
    ft.main()
    printed = capsys.readouterr().out
    assert "cnn" in printed and "skipped" in printed
    for model in ("metadata_baseline", "classical_ml"):
        used = ft.load_threshold(tmp_path / "models" / f"{model}_threshold.json")
        recorded = ft.load_threshold(reports / model / "threshold.json")
        assert used == recorded and used["model"] == model


def test_main_exits_when_nothing_can_be_frozen(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["freeze", "--reports-dir", str(tmp_path / "none"),
                                     "--models-dir", str(tmp_path / "models")])
    with pytest.raises(SystemExit):
        ft.main()


# ---------------------------------------------------------------- use in predict

def test_predict_reports_the_flag(tmp_path):
    rng = np.random.default_rng(0)
    models = tmp_path / "models"
    models.mkdir()
    features = pd.DataFrame(rng.normal(size=(120, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    joblib.dump(classical_ml.build_pipeline().fit(features, rng.integers(0, 2, 120)), models / "classical_ml.joblib")
    image = Image.new("RGB", (600, 450), (215, 170, 150))
    ImageDraw.Draw(image).ellipse((200, 130, 400, 320), fill=(90, 50, 35))
    image.save(tmp_path / "lesion.jpg")

    without = predict.predict_image(tmp_path / "lesion.jpg", predict.load_models(models, tmp_path / "reports"))
    assert without["classical_ml"]["flagged"] is None
    raw = without["classical_ml"]["raw"]

    for threshold_raw, expected in ((raw, True), (np.nextafter(raw, 1.0), False), (0.0, True), (1.0, False)):
        save_calibrator({"threshold_raw": threshold_raw, "threshold_recalibrated": threshold_raw / 2},
                        models / "classical_ml_threshold.json")
        result = predict.predict_image(tmp_path / "lesion.jpg", predict.load_models(models, tmp_path / "reports"))
        assert result["classical_ml"]["flagged"] is expected
        assert result["classical_ml"]["threshold_raw"] == threshold_raw

    text = predict.format_results(result, {})
    assert "flagged" in text and "no" in text
