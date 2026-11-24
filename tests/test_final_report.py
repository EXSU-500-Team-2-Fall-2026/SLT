"""Analysis of frozen-model predictions: metrics, the paired group
bootstrap, the verdict rule and the files written. Synthetic predictions
only; no model or image is needed."""
import json

import numpy as np
import pandas as pd
import pytest

from src.eval import final_report as fr
from src.eval.recalibration import apply_recalibration, fit_logistic_recalibration, save_calibrator


def _predictions(n_groups=300, seed=0, models=("metadata_baseline", "classical_ml", "cnn")):
    """Two images per group. Raw probabilities are shifted upwards (too
    high); the recalibrated column undoes the shift. Later models rank
    better."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        risk = rng.normal(-1.5, 1.2)
        label = int(rng.random() < 1 / (1 + np.exp(-risk)))
        dx = rng.choice(["mel", "bcc", "akiec"]) if label else rng.choice(["nv", "bkl"])
        for image in range(2):
            rows.append({"image_id": f"ISIC_{g:04d}_{image}", "lesion_id": f"L{g}", "group_id": f"G{g}", "dx": dx,
                         "label": label, "split": "tune", "dataset": "site_a" if g % 2 else "site_b", "risk": risk})
    table = pd.DataFrame(rows)
    for position, model in enumerate(models):
        noise = rng.normal(0, 1.6 - 0.5 * position, len(table))
        logit = table["risk"] + noise
        table[f"{model}_raw"] = 1 / (1 + np.exp(-(logit + 1.5)))
        table[f"{model}_recalibrated"] = 1 / (1 + np.exp(-logit))
        table[f"{model}_flagged"] = (table[f"{model}_raw"] >= 0.5).astype(int)
    return table.drop(columns="risk")


def test_metrics_for_one_model():
    table = _predictions()
    y = table["label"].to_numpy()
    m = fr.model_metrics(y, table["cnn_raw"].to_numpy(), table["cnn_recalibrated"].to_numpy(),
                         table["cnn_flagged"].to_numpy())
    assert m["auroc"] == pytest.approx(m["auroc_recalibrated"])        # a monotone mapping keeps the ranking
    assert m["auprc"] == pytest.approx(m["auprc_recalibrated"])
    assert m["brier_difference"] == pytest.approx(m["brier_recalibrated"] - m["brier_original"])
    assert m["brier_difference"] < 0 and m["ece_difference"] < 0       # undoing the shift helps
    assert m["calibration_intercept_original"] < -1 < m["calibration_intercept_recalibrated"]
    flagged = table["cnn_flagged"].to_numpy().astype(bool)
    assert m["sensitivity"] == pytest.approx(flagged[y == 1].mean())
    assert m["specificity"] == pytest.approx((~flagged[y == 0]).mean())


def test_all_metrics_includes_every_pair_of_models():
    metrics = fr.all_metrics(_predictions(), list(fr.MODEL_NAMES))
    pairs = {"classical_ml_minus_metadata_baseline", "cnn_minus_metadata_baseline", "cnn_minus_classical_ml"}
    assert pairs <= set(metrics)
    assert metrics["cnn_minus_classical_ml"]["auroc"] == pytest.approx(
        metrics["cnn"]["auroc"] - metrics["classical_ml"]["auroc"])
    assert set(metrics["cnn_minus_classical_ml"]) == set(fr.PAIRED_METRICS)


def test_group_bootstrap_is_reproducible_and_brackets_the_estimates():
    table = _predictions()
    models = list(fr.MODEL_NAMES)
    first = fr.group_bootstrap(table, models, n_boot=150, seed=1)
    again = fr.group_bootstrap(table, models, n_boot=150, seed=1)
    assert first == again and first["_n_boot"] == 150
    estimates = fr.all_metrics(table, models)
    for metric in ("auroc", "sensitivity", "specificity", "brier_original", "brier_difference"):
        low, high = first["cnn"][metric]
        assert low <= estimates["cnn"][metric] <= high


def test_group_bootstrap_resamples_whole_groups():
    """With every image of a group identical, a group bootstrap gives the
    same spread whether a group has 1 image or 5; an image bootstrap would
    not. Here: intervals do not shrink when each group's images are repeated."""
    base = _predictions(n_groups=200, seed=3, models=("cnn",))
    one_image = base.drop_duplicates("group_id").reset_index(drop=True)
    repeated = pd.concat([one_image] * 5, ignore_index=True)
    repeated["image_id"] = [f"IMG_{i}" for i in range(len(repeated))]

    narrow = fr.group_bootstrap(one_image, ["cnn"], n_boot=300, seed=5)["cnn"]["auroc"]
    wide = fr.group_bootstrap(repeated, ["cnn"], n_boot=300, seed=5)["cnn"]["auroc"]
    assert narrow == pytest.approx(wide)  # same groups drawn, same statistic: clustering fully respected


def test_group_bootstrap_uses_the_same_groups_for_every_model():
    """Two identical models must have a difference of exactly zero in every resample."""
    table = _predictions(models=("classical_ml", "cnn"))
    for suffix in ("raw", "recalibrated", "flagged"):
        table[f"cnn_{suffix}"] = table[f"classical_ml_{suffix}"]
    intervals = fr.group_bootstrap(table, ["classical_ml", "cnn"], n_boot=100, seed=2)
    for metric in fr.PAIRED_METRICS:
        assert intervals["cnn_minus_classical_ml"][metric] == [0.0, 0.0]


@pytest.mark.parametrize("interval, expected", [
    ([-0.05, -0.01], "improvement"), ([0.01, 0.05], "deterioration"),
    ([-0.02, 0.01], "no clear evidence of change"), ([-0.02, 0.0], "no clear evidence of change"),
])
def test_verdict_rule(interval, expected):
    assert fr.verdict(interval) == expected


def test_wilson_interval():
    low, high = fr.wilson_interval(9, 10)
    assert 0.59 < low < 0.60 and 0.98 < high < 0.99
    assert fr.wilson_interval(0, 10)[0] == pytest.approx(0.0) and fr.wilson_interval(10, 10)[1] == pytest.approx(1.0)
    assert all(np.isnan(fr.wilson_interval(0, 0)))


def test_flag_rate_by_diagnosis():
    table = _predictions()
    by_dx = fr.flag_rate_by_diagnosis(table, ["cnn"])
    assert set(by_dx["dx"]) == set(table["dx"])
    mel = by_dx[by_dx["dx"] == "mel"].iloc[0]
    part = table[table["dx"] == "mel"]
    assert mel["group"] == "malignant/premalignant" and mel["images"] == len(part)
    assert mel["flag_rate"] == pytest.approx(part["cnn_flagged"].mean())
    assert mel["wilson_low"] <= mel["flag_rate"] <= mel["wilson_high"]
    assert by_dx[by_dx["dx"] == "nv"].iloc[0]["group"] == "benign"


def test_performance_by_source():
    table = _predictions()
    by_source = fr.performance_by_source(table, ["cnn"])
    assert set(by_source["source"]) == {"site_a", "site_b"}
    assert by_source["images"].sum() == len(table)
    assert fr.performance_by_source(table.drop(columns="dataset"), ["cnn"]).empty


def test_write_report_files_and_contents(tmp_path):
    report = fr.write_report(_predictions(), tmp_path / "out", n_boot=60, seed=4)
    for name in ("metrics.json", "summary.md", "flag_rate_by_diagnosis.csv", "performance_by_source.csv",
                 "reliability_before_after.png"):
        assert (tmp_path / "out" / name).stat().st_size > 0
    saved = json.load(open(tmp_path / "out" / "metrics.json"))
    assert saved == report
    assert saved["cohort"] == {"split": "tune", "images": 600, "lesions": 300, "groups": 300,
                               "positive_images": int(_predictions()["label"].sum())}
    assert saved["bootstrap"]["resampling_unit"] == "group_id" and saved["bootstrap"]["n_boot_used"] == 60
    assert saved["models"]["cnn"]["recalibration_verdict"] == "improvement"
    summary = (tmp_path / "out" / "summary.md").read_text()
    assert "Difference (primary)" in summary and "CNN minus Classical ML" in summary and "| mel |" in summary


def test_assemble_from_saved_predictions(tmp_path):
    rng = np.random.default_rng(0)
    splits, reports = tmp_path / "splits", tmp_path / "reports"
    splits.mkdir()
    y = rng.integers(0, 2, 80)
    manifest = pd.DataFrame({"image_id": [f"I{i}" for i in range(80)], "lesion_id": [f"L{i // 2}" for i in range(80)],
                             "group_id": [f"G{i // 2}" for i in range(80)], "dx": np.where(y == 1, "mel", "nv"),
                             "label": y, "split": "tune"})
    manifest.to_csv(splits / "tune.csv", index=False)
    raw = np.clip(0.3 * y + rng.uniform(0.05, 0.65, 80), 0.01, 0.99)
    (reports / "cnn").mkdir(parents=True)
    saved = manifest[["image_id", "lesion_id", "group_id", "split", "label"]].assign(prob=raw)
    saved.sample(frac=1, random_state=0).to_csv(reports / "cnn" / "predictions_tune.csv", index=False)  # other order
    calibrator = fit_logistic_recalibration(y, raw)
    save_calibrator(calibrator, reports / "cnn" / "calibrator.json")
    save_calibrator({"threshold_raw": 0.4, "threshold_recalibrated": 0.2}, reports / "cnn" / "threshold.json")

    table = fr.assemble_from_saved(reports, "tune", splits)
    assert fr.models_in(table) == ["cnn"]
    assert list(table["image_id"]) == list(manifest["image_id"])          # manifest order, matched by image_id
    assert np.allclose(table["cnn_raw"], raw)
    assert np.allclose(table["cnn_recalibrated"], apply_recalibration(raw, calibrator))
    assert list(table["cnn_flagged"]) == list((raw >= 0.4).astype(int))


def test_assemble_refuses_the_held_out_split(tmp_path):
    with pytest.raises(ValueError, match="score_heldout"):
        fr.assemble_from_saved(tmp_path, "test", tmp_path)
