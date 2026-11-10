"""Fit one logistic recalibration mapping per trained model.

Step 3 of the "Future recalibration protocol" in
reports/experimental-design.md. Each mapping is fitted on that model's
saved predictions for the calibration split
(<reports-dir>/<model>/predictions_calibration.csv, written by the training
scripts with --predict-splits) and on nothing else. See
src/eval/recalibration.py for the method and its numerical settings.

For each model the fitted coefficients are written to

    <models-dir>/<model>_calibrator.json    used by src/predict.py
    <reports-dir>/<model>/calibrator.json   the committed record

This script refuses to fit on the held-out test set, or on the split that
was used to select the model (the eval split recorded in metrics.json):
calibration cases must not have influenced model fitting or selection.

It deliberately prints no before/after metrics. A mapping always looks
good on the cases it was fitted to; whether recalibration helps is
assessed once, on the held-out cohort.

If a model is retrained, its calibrator is out of date and must be refitted.

Usage:
    python -m src.eval.fit_calibrators --reports-dir reports/v2 --models-dir models
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.eval.recalibration import fit_logistic_recalibration, save_calibrator
from src.eval.reporting import HELD_OUT_SPLIT

MODEL_NAMES = ("metadata_baseline", "classical_ml", "cnn")


def fit_model_calibrator(model: str, reports_dir: Path, split: str = "calibration") -> dict:
    """Fit from <reports_dir>/<model>/predictions_<split>.csv. Raises
    FileNotFoundError if the predictions are missing and ValueError if they
    must not be used for calibration."""
    if split == HELD_OUT_SPLIT:
        raise ValueError("calibrators must not be fitted on the held-out test set")

    model_dir = Path(reports_dir) / model
    predictions_path = model_dir / f"predictions_{split}.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)

    metrics_path = model_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            selection_split = json.load(f).get("eval_split")
        if selection_split == split:
            raise ValueError(
                f"{model}: '{split}' is the split the model was selected on; "
                "fit the calibrator on a split that played no part in training or selection"
            )

    predictions = pd.read_csv(predictions_path)
    other = set(predictions["split"].unique()) - {split}
    if other:
        raise ValueError(f"{predictions_path} contains rows from other splits: {sorted(other)}")

    calibrator = fit_logistic_recalibration(predictions["label"], predictions["prob"])
    calibrator.update({"model": model, "fitted_on": split, "predictions_file": predictions_path.as_posix()})
    return calibrator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/v2"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    args = parser.parse_args()

    fitted = 0
    for model in args.models:
        try:
            calibrator = fit_model_calibrator(model, args.reports_dir, args.split)
        except FileNotFoundError as missing:
            print(f"{model:<20} skipped: {missing} not found (train with --predict-splits {args.split})")
            continue
        except ValueError as problem:
            sys.exit(f"Not fitted: {problem}")

        save_calibrator(calibrator, args.models_dir / f"{model}_calibrator.json")
        save_calibrator(calibrator, args.reports_dir / model / "calibrator.json")
        fitted += 1
        note = "" if calibrator["converged"] else "  (WARNING: the fit did not converge)"
        print(
            f"{model:<20} intercept {calibrator['intercept']:+.3f}  slope {calibrator['slope']:.3f}  "
            f"n={calibrator['n_calibration']} ({calibrator['n_positive']} positive){note}"
        )

    if fitted == 0:
        sys.exit("No calibrators were fitted.")


if __name__ == "__main__":
    main()
