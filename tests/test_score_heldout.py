"""The one-time held-out scoring step, exercised end to end on a small
synthetic project with the two models that need no PyTorch: frozen-state
checks, cohort checks, the rehearsal comparison, the outputs and the lock."""
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

from src.eval import final_report, fit_calibrators, freeze_thresholds, score_heldout
from src.features.handcrafted import FEATURE_NAMES
from src.models import classical_ml, metadata_baseline

SIZES = {"train": 160, "tune": 60, "calibration": 60, "test": 40}
TWO_MODELS = ["metadata_baseline", "classical_ml"]


def _lesion_image(path, rng):
    shade = int(rng.integers(40, 120))
    image = Image.new("RGB", (120, 90), (215, 170, 150))
    ImageDraw.Draw(image).ellipse((35, 20, 85, 70), fill=(shade, shade // 2, shade // 3))
    image.save(path)


@pytest.fixture()
def project(tmp_path, monkeypatch):
    """Manifests, metadata tables, cached development features, images for
    the tune and test subsets, and two trained, calibrated, thresholded
    models."""
    rng = np.random.default_rng(0)
    raw = tmp_path / "raw"
    for folder in ("splits", "features", "raw/HAM10000_images", "raw/ISIC2018_Task3_Test_Images"):
        (tmp_path / folder).mkdir(parents=True)

    frames, counter = {}, 0
    for split, n in SIZES.items():
        rows = []
        for i in range(n):
            counter += 1
            label = int(i % 3 == 0)
            rows.append({"image_id": f"ISIC_{counter:07d}", "lesion_id": f"L{counter // 2}", "group_id": f"G{counter // 2}",
                         "dx": "mel" if label else "nv", "label": label, "split": split})
        frames[split] = pd.DataFrame(rows)
        frames[split].to_csv(tmp_path / "splits" / f"{split}.csv", index=False)

    def metadata_for(frame):
        return pd.DataFrame({
            "lesion_id": frame["lesion_id"], "image_id": frame["image_id"], "dx": frame["dx"],
            "age": np.where(frame["label"] == 1, 65.0, 40.0) + rng.integers(-15, 15, len(frame)),
            "sex": rng.choice(["male", "female"], len(frame)),
            "localization": rng.choice(["back", "face", "trunk"], len(frame)),
            "dataset": rng.choice(["site_a", "site_b"], len(frame)),
        })

    development = pd.concat([frames[s] for s in ("train", "tune", "calibration")], ignore_index=True)
    metadata_for(development).to_csv(raw / "HAM10000_metadata.tab", index=False)
    truth = metadata_for(frames["test"])
    truth.loc[:4, "age"] = np.nan          # missing metadata in the held-out table
    truth.loc[:2, "sex"] = np.nan
    truth.to_csv(raw / "ISIC2018_Task3_Test_GroundTruth.tab", sep="\t", index=False)

    features = pd.DataFrame(rng.normal(size=(len(development), len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    features["asymmetry"] += 1.5 * development["label"]
    features.insert(0, "label", development["label"])
    features.insert(0, "image_id", development["image_id"])
    features.to_csv(tmp_path / "features" / "train.csv", index=False)

    for image_id in frames["tune"]["image_id"]:
        _lesion_image(raw / "HAM10000_images" / f"{image_id}.jpg", rng)
    for image_id in frames["test"]["image_id"][:-1]:   # the last test image has no file (withdrawn)
        _lesion_image(raw / "ISIC2018_Task3_Test_Images" / f"{image_id}.jpg", rng)

    reports, models = tmp_path / "reports", tmp_path / "models"
    for module, name, extra in ((metadata_baseline, "metadata_baseline", ["--raw-dir", str(raw)]),
                                (classical_ml, "classical_ml", ["--features-dir", str(tmp_path / "features")])):
        monkeypatch.setattr("sys.argv", [name, "--splits-dir", str(tmp_path / "splits"), "--eval-split", "tune",
                                         "--predict-splits", "tune", "calibration", "--n-boot", "20",
                                         "--out", str(models / f"{name}.joblib"),
                                         "--reports-dir", str(reports / name), *extra])
        module.main()
    monkeypatch.setattr("sys.argv", ["fit", "--reports-dir", str(reports), "--models-dir", str(models)])
    fit_calibrators.main()
    monkeypatch.setattr("sys.argv", ["freeze", "--reports-dir", str(reports), "--models-dir", str(models)])
    freeze_thresholds.main()
    return tmp_path


def _argv(project, *extra):
    return ["score", "--splits-dir", str(project / "splits"), "--raw-dir", str(project / "raw"),
            "--models-dir", str(project / "models"), "--reports-dir", str(project / "reports"),
            "--features-dir", str(project / "features"), "--models", *TWO_MODELS, *extra]


def test_check_frozen_passes_then_catches_a_changed_calibrator(project):
    record, problems = score_heldout.check_frozen(TWO_MODELS, project / "models", project / "reports")
    assert problems == []
    assert len(record["classical_ml"]["model_sha256"]) == 64
    assert {"calibrator", "threshold"} <= set(record["classical_ml"])

    path = project / "models" / "classical_ml_calibrator.json"
    changed = json.load(open(path))
    changed["intercept"] += 0.5
    json.dump(changed, open(path, "w"))
    _, problems = score_heldout.check_frozen(TWO_MODELS, project / "models", project / "reports")
    assert len(problems) == 1 and "differs from the committed record" in problems[0]


def test_check_frozen_reports_missing_pieces(project):
    (project / "models" / "metadata_baseline_threshold.json").rename(project / "models" / "elsewhere.json")
    _, problems = score_heldout.check_frozen(TWO_MODELS + ["cnn"], project / "models", project / "reports")
    assert any("metadata_baseline: threshold missing" in p for p in problems)
    assert any("cnn: model file" in p for p in problems)


def test_cohort_excludes_images_without_a_file(project):
    cohort, image_dir, excluded = score_heldout.load_cohort("test", project / "splits", project / "raw")
    assert len(cohort) == 39 and len(excluded) == 1 and image_dir.name == "ISIC2018_Task3_Test_Images"
    assert {"age", "sex", "localization", "dataset", "group_id"} <= set(cohort.columns)
    assert cohort["age"].isna().sum() == 5


def test_cohort_check_catches_a_diagnosis_mismatch(project):
    path = project / "raw" / "ISIC2018_Task3_Test_GroundTruth.tab"
    truth = pd.read_csv(path, sep="\t")
    truth.loc[3, "dx"] = "bcc" if truth.loc[3, "dx"] != "bcc" else "nv"
    truth.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="diagnosis differs"):
        score_heldout.load_cohort("test", project / "splits", project / "raw")


def test_rehearsal_reproduces_saved_predictions(project, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", _argv(project, "--rehearsal"))
    score_heldout.main()
    assert "Rehearsal passed" in capsys.readouterr().out
    out = project / "reports" / "final_rehearsal"
    record = json.load(open(out / "run_record.json"))
    assert record["split"] == "tune" and record["scored_images"] == 60
    assert max(record["largest_difference_from_saved_predictions"].values()) < 1e-9
    assert not (project / "reports" / "final").exists()   # the rehearsal does not touch held-out outputs


def test_rehearsal_fails_when_scoring_does_not_match(project, monkeypatch):
    path = project / "reports" / "classical_ml" / "predictions_tune.csv"
    saved = pd.read_csv(path)
    saved["prob"] = (saved["prob"] * 0.5).clip(0.001, 0.999)
    saved.to_csv(path, index=False)
    monkeypatch.setattr("sys.argv", _argv(project, "--rehearsal"))
    with pytest.raises(SystemExit, match="Rehearsal FAILED"):
        score_heldout.main()


def test_held_out_run_writes_predictions_once(project, monkeypatch):
    monkeypatch.setattr("sys.argv", _argv(project))
    score_heldout.main()
    out = project / "reports" / "final"
    predictions = pd.read_csv(out / "predictions_test.csv")
    manifest = pd.read_csv(project / "splits" / "test.csv")

    assert list(predictions["image_id"]) == list(manifest["image_id"][:-1])
    assert (predictions["split"] == "test").all() and "dataset" in predictions.columns
    for model in TWO_MODELS:
        threshold = json.load(open(project / "reports" / model / "threshold.json"))["threshold_raw"]
        assert predictions[f"{model}_raw"].between(0, 1).all() and predictions[f"{model}_recalibrated"].between(0, 1).all()
        assert list(predictions[f"{model}_flagged"]) == list((predictions[f"{model}_raw"] >= threshold).astype(int))

    record = json.load(open(out / "run_record.json"))
    assert record["purpose"] == "held-out evaluation"
    assert record["manifest_images"] == 40 and record["scored_images"] == 39
    assert record["excluded_images"][0]["image_id"] == manifest["image_id"].iloc[-1]
    assert record["missing_metadata"] == {"age": 5, "sex": 3, "localization": 0}
    assert set(record["frozen"]) == set(TWO_MODELS)
    features = pd.read_csv(out / "features_test.csv")
    assert len(features) == 39 and list(features.columns) == ["image_id", *FEATURE_NAMES]

    # the analysis runs on the file that was written
    report = final_report.write_report(predictions, out, n_boot=40)
    assert report["cohort"]["images"] == 39 and set(report["models"]) == set(TWO_MODELS)

    # a second scoring run is refused and changes nothing
    before = (out / "predictions_test.csv").read_bytes()
    monkeypatch.setattr("sys.argv", _argv(project))
    with pytest.raises(SystemExit, match="scored once"):
        score_heldout.main()
    assert (out / "predictions_test.csv").read_bytes() == before


def test_held_out_run_refuses_when_not_frozen(project, monkeypatch):
    (project / "models" / "classical_ml_threshold.json").rename(project / "models" / "elsewhere.json")
    monkeypatch.setattr("sys.argv", _argv(project))
    with pytest.raises(SystemExit, match="Not everything is frozen"):
        score_heldout.main()
    assert not (project / "reports" / "final").exists()
