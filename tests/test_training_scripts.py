"""Models 1 and 2 run end to end on small synthetic data: split options,
saved predictions, the test-set guard, the feature lookup, and fitting
calibrators from the saved predictions."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.eval import fit_calibrators, reporting
from src.eval.metrics import calibration_metrics, discrimination_metrics
from src.features.handcrafted import FEATURE_NAMES
from src.models import classical_ml, metadata_baseline

SIZES = {"train": 160, "val": 60, "tune": 60, "calibration": 60, "test": 40}


@pytest.fixture()
def project(tmp_path):
    """Manifests, a metadata table and cached feature tables for fake images."""
    rng = np.random.default_rng(0)
    (tmp_path / "splits").mkdir()
    (tmp_path / "raw").mkdir()
    (tmp_path / "features").mkdir()
    manifests, counter = [], 0
    for split, n in SIZES.items():
        rows = []
        for i in range(n):
            counter += 1
            rows.append({"image_id": f"ISIC_{counter:07d}", "lesion_id": f"L{counter // 2}",
                         "group_id": f"G{counter // 2}", "dx": "mel" if i % 3 == 0 else "nv",
                         "label": int(i % 3 == 0), "split": split})
        frame = pd.DataFrame(rows)
        frame.to_csv(tmp_path / "splits" / f"{split}.csv", index=False)
        manifests.append(frame)
    everything = pd.concat(manifests, ignore_index=True)

    metadata = pd.DataFrame({
        "lesion_id": everything["lesion_id"], "image_id": everything["image_id"], "dx": everything["dx"],
        "age": np.where(everything["label"] == 1, 65, 40) + rng.integers(-15, 15, len(everything)),
        "sex": rng.choice(["male", "female"], len(everything)),
        "localization": rng.choice(["back", "face", "trunk"], len(everything)),
    })
    metadata.to_csv(tmp_path / "raw" / "HAM10000_metadata.tab", index=False)

    features = pd.DataFrame(rng.normal(size=(len(everything), len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    features["asymmetry"] += 1.5 * everything["label"]
    features.insert(0, "label", everything["label"])
    features.insert(0, "image_id", everything["image_id"])
    development = features[everything["split"] != "test"]
    development.iloc[:200].to_csv(tmp_path / "features" / "train.csv", index=False)
    development.iloc[200:].to_csv(tmp_path / "features" / "val.csv", index=False)
    return tmp_path


def _run(module, monkeypatch, project, name, *extra):
    argv = [name, "--splits-dir", str(project / "splits"), "--out", str(project / "models" / f"{name}.joblib"),
            "--reports-dir", str(project / "reports" / name), "--n-boot", "20", *extra]
    argv += ["--raw-dir", str(project / "raw")] if name == "metadata_baseline" else \
            ["--features-dir", str(project / "features")]
    monkeypatch.setattr("sys.argv", argv)
    module.main()
    with open(project / "reports" / name / "metrics.json") as f:
        return json.load(f)


MODELS = [(metadata_baseline, "metadata_baseline"), (classical_ml, "classical_ml")]


@pytest.mark.parametrize("module, name", MODELS)
def test_default_run_reports_on_val(module, name, monkeypatch, project):
    report = _run(module, monkeypatch, project, name)
    assert report["eval_split"] == "val" and report["n_train"] == 160 and report["n_val"] == 60
    assert {"discrimination", "calibration", "bootstrap"} <= set(report)
    assert not list((project / "reports" / name).glob("predictions_*.csv"))


@pytest.mark.parametrize("module, name", MODELS)
def test_three_way_split_saves_predictions(module, name, monkeypatch, project):
    report = _run(module, monkeypatch, project, name, "--eval-split", "tune", "--predict-splits", "tune", "calibration")
    assert report["eval_split"] == "tune" and report["n_val"] == 60

    for split in ("tune", "calibration"):
        predictions = pd.read_csv(project / "reports" / name / f"predictions_{split}.csv")
        manifest = pd.read_csv(project / "splits" / f"{split}.csv")
        assert list(predictions.columns) == ["image_id", "lesion_id", "group_id", "split", "label", "prob"]
        assert list(predictions["image_id"]) == list(manifest["image_id"])
        assert list(predictions["label"]) == list(manifest["label"])
        assert (predictions["split"] == split).all() and predictions["prob"].between(0, 1).all()

    # the reported AUROC is the AUROC of the saved tune predictions
    tune = pd.read_csv(project / "reports" / name / "predictions_tune.csv")
    assert report["discrimination"]["auroc"] == pytest.approx(discrimination_metrics(tune["label"], tune["prob"]).auroc)


@pytest.mark.parametrize("module, name", MODELS)
@pytest.mark.parametrize("extra", [["--eval-split", "test"], ["--predict-splits", "calibration", "test"]])
def test_held_out_test_set_is_refused(module, name, extra, monkeypatch, project):
    with pytest.raises(SystemExit):
        _run(module, monkeypatch, project, name, *extra)
    assert not (project / "models").exists()


def test_classical_features_follow_the_manifest_not_the_cache_files(monkeypatch, project):
    """The cached tables are split differently from the manifests on purpose."""
    pool = classical_ml.load_feature_pool(project / "features")
    tune = classical_ml.load_split_features(project / "splits" / "tune.csv", pool)
    manifest = pd.read_csv(project / "splits" / "tune.csv")
    assert list(tune["image_id"]) == list(manifest["image_id"])
    assert tune[FEATURE_NAMES].notna().all().all()


def test_classical_reports_images_without_cached_features(project):
    pool = classical_ml.load_feature_pool(project / "features")
    with pytest.raises(ValueError, match="no cached features"):
        classical_ml.load_split_features(project / "splits" / "test.csv", pool)


def test_saved_model_reproduces_saved_predictions(monkeypatch, project):
    _run(classical_ml, monkeypatch, project, "classical_ml", "--eval-split", "tune", "--predict-splits", "calibration")
    model = joblib.load(project / "models" / "classical_ml.joblib")
    pool = classical_ml.load_feature_pool(project / "features")
    calibration = classical_ml.load_split_features(project / "splits" / "calibration.csv", pool)
    saved = pd.read_csv(project / "reports" / "classical_ml" / "predictions_calibration.csv")
    assert np.allclose(model.predict_proba(calibration[FEATURE_NAMES])[:, 1], saved["prob"])


# ---------------------------------------------------------------- reporting helpers

def test_evaluate_matches_the_metric_functions():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = np.clip(0.3 * y + rng.uniform(0, 0.7, 200), 0, 1)
    sections, calib = reporting.evaluate(y, p, threshold=0.4, n_boot=20)
    assert sections["discrimination"]["auroc"] == discrimination_metrics(y, p, 0.4).auroc
    assert sections["discrimination"]["threshold"] == 0.4
    assert sections["calibration"]["brier_score"] == calibration_metrics(y, p).brier_score
    assert sections["bootstrap"]["n_boot"] <= 20
    assert calib.ece == sections["calibration"]["ece"]


def test_write_predictions_checks_length(tmp_path):
    frame = pd.DataFrame({"image_id": ["a", "b"], "lesion_id": ["l", "l"], "label": [0, 1]})
    with pytest.raises(ValueError):
        reporting.write_predictions(frame, [0.1], "tune", tmp_path / "p.csv")
    out = reporting.write_predictions(frame, [0.1, 0.9], "tune", tmp_path / "p.csv")
    assert list(out.columns) == ["image_id", "lesion_id", "split", "label", "prob"]  # no group_id in an old manifest


# ---------------------------------------------------------------- calibrators

def _trained(monkeypatch, project):
    for module, name in MODELS:
        _run(module, monkeypatch, project, name, "--eval-split", "tune", "--predict-splits", "tune", "calibration")


def test_fit_calibrators_writes_both_copies(monkeypatch, project, capsys):
    _trained(monkeypatch, project)
    monkeypatch.setattr("sys.argv", ["fit", "--reports-dir", str(project / "reports"),
                                     "--models-dir", str(project / "models")])
    fit_calibrators.main()
    printed = capsys.readouterr().out
    assert "cnn" in printed and "skipped" in printed  # no CNN predictions in this test

    for name in ("metadata_baseline", "classical_ml"):
        with open(project / "models" / f"{name}_calibrator.json") as f:
            used = json.load(f)
        with open(project / "reports" / name / "calibrator.json") as f:
            recorded = json.load(f)
        assert used == recorded
        assert used["fitted_on"] == "calibration" and used["n_calibration"] == 60 and used["model"] == name
        assert np.isfinite(used["intercept"]) and np.isfinite(used["slope"])


def test_calibrator_refuses_the_selection_split(monkeypatch, project):
    _trained(monkeypatch, project)
    with pytest.raises(ValueError, match="selected on"):
        fit_calibrators.fit_model_calibrator("classical_ml", project / "reports", split="tune")


def test_calibrator_refuses_the_test_set(project):
    with pytest.raises(ValueError, match="held-out test set"):
        fit_calibrators.fit_model_calibrator("classical_ml", project / "reports", split="test")


def test_calibrator_refuses_mixed_predictions(monkeypatch, project):
    _trained(monkeypatch, project)
    path = project / "reports" / "classical_ml" / "predictions_calibration.csv"
    mixed = pd.read_csv(path)
    mixed.loc[0, "split"] = "train"
    mixed.to_csv(path, index=False)
    with pytest.raises(ValueError, match="other splits"):
        fit_calibrators.fit_model_calibrator("classical_ml", project / "reports")


def test_fit_calibrators_exits_when_nothing_to_fit(monkeypatch, project):
    monkeypatch.setattr("sys.argv", ["fit", "--reports-dir", str(project / "reports"),
                                     "--models-dir", str(project / "models")])
    with pytest.raises(SystemExit):
        fit_calibrators.main()
