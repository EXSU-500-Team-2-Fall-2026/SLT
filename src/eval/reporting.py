"""Shared pieces of the three training scripts: the evaluation block that
goes into metrics.json, saving predictions with their identifiers, and the
rule that keeps the held-out test set out of training-time scripts.

Predictions are saved with image, lesion and (when the manifest has one)
connected-group identifiers, so that later steps can fit calibrators,
bootstrap by lesion group and compare models on the same cases without
re-running a model.
"""
from pathlib import Path

import pandas as pd

from src.eval.metrics import CalibrationMetrics, bootstrap_metrics_ci, calibration_metrics, discrimination_metrics

ID_COLUMNS = ["image_id", "lesion_id", "group_id"]
HELD_OUT_SPLIT = "test"


def split_label(split: str) -> str:
    return {"val": "validation", "tune": "tuning"}.get(split, split)


def evaluate(y_true, y_prob, threshold: float = 0.5, n_boot: int = 1000) -> tuple[dict, CalibrationMetrics]:
    """Return (the discrimination / calibration / bootstrap sections of
    metrics.json, the CalibrationMetrics needed for the reliability diagram)."""
    disc = discrimination_metrics(y_true, y_prob, threshold=threshold)
    calib = calibration_metrics(y_true, y_prob)
    ci = bootstrap_metrics_ci(y_true, y_prob, threshold=threshold, n_boot=n_boot)
    sections = {
        "discrimination": {
            "auroc": disc.auroc,
            "auroc_ci95": ci["auroc_ci"],
            "auprc": disc.auprc,
            "auprc_ci95": ci["auprc_ci"],
            "sensitivity": disc.sensitivity,
            "sensitivity_ci95": ci["sensitivity_ci"],
            "specificity": disc.specificity,
            "specificity_ci95": ci["specificity_ci"],
            "threshold": disc.threshold,
        },
        "calibration": {
            "brier_score": calib.brier_score,
            "brier_score_ci95": ci["brier_score_ci"],
            "ece": calib.ece,
            "calibration_slope": calib.calibration_slope,
            "calibration_intercept": calib.calibration_intercept,
        },
        "bootstrap": {"n_boot": ci["n_boot"], "ci_level": ci["ci_level"]},
    }
    return sections, calib


def write_predictions(split_df: pd.DataFrame, y_prob, split: str, out_path: Path) -> pd.DataFrame:
    """Save one row per image: identifiers, split name, label and the raw
    (not recalibrated) predicted probability."""
    if len(split_df) != len(y_prob):
        raise ValueError(f"{len(split_df)} rows but {len(y_prob)} predictions")
    columns = [c for c in ID_COLUMNS if c in split_df.columns]
    out = split_df[columns].reset_index(drop=True).copy()
    out["split"] = split
    out["label"] = split_df["label"].to_numpy().astype(int)
    out["prob"] = pd.Series(y_prob, dtype=float).to_numpy()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def check_development_splits(parser, *split_names: str) -> None:
    """Stop with a usage error if a training-time script is pointed at the
    held-out test set. That set is scored once, by the final evaluation,
    after models, calibrators and thresholds are frozen."""
    if HELD_OUT_SPLIT in split_names:
        parser.error(
            "the held-out test set cannot be used here: it is evaluated once, by the final "
            "evaluation script, after models, calibrators and thresholds are frozen"
        )


def add_split_arguments(parser) -> None:
    parser.add_argument(
        "--eval-split", default="val",
        help="development split to report on and select with (default: val; use 'tune' with data/splits_v2)",
    )
    parser.add_argument(
        "--predict-splits", nargs="*", default=[],
        help="development splits whose predictions are saved to <reports-dir>/predictions_<split>.csv "
             "(e.g. tune calibration)",
    )
    parser.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for the confidence intervals")
