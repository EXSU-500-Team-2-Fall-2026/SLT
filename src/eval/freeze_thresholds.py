"""Choose and freeze one decision threshold per model, on the tuning subset.

Rule, fixed before any held-out result exists: for each model, the highest
threshold at which sensitivity on the tuning subset is at least the target
(default 0.90). A lesion is flagged when its score is at or above the
threshold.

Why a fixed sensitivity: the motivating use is triage, where a missed
malignant lesion is the costlier error, so sensitivity is set in advance
and the models are compared on the specificity they reach at that
sensitivity. The 0.5 threshold used for earlier development reporting is a
convention; after recalibration, which lowers the predicted probabilities,
0.5 would flag very few lesions.

The threshold is stored on two scales that flag exactly the same cases,
because recalibration with a positive slope does not change the ranking:

- threshold_raw: on the model's raw probability. Decisions are made on
  this scale, so they do not depend on the calibrator.
- threshold_recalibrated: the same point after recalibration, for display.

For each model the result is written to

    <models-dir>/<model>_threshold.json    used by src/predict.py
    <reports-dir>/<model>/threshold.json   the committed record

Limits to state wherever these thresholds are reported:
- Sensitivity on the tuning subset meets the target by construction. On
  new cases it will be lower or higher by chance; that is what the
  held-out evaluation measures.
- The tuning subset has few positive cases (136 in the committed split),
  so the threshold itself is uncertain.
- The target is a study convention, not a clinically validated operating
  point.

The script refuses the held-out test set and the calibration subset. If a
model is retrained, its threshold is out of date and must be frozen again.

Usage:
    python -m src.eval.freeze_thresholds --reports-dir reports/v2 --models-dir models
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.recalibration import apply_recalibration, load_calibrator, save_calibrator
from src.eval.reporting import HELD_OUT_SPLIT

MODEL_NAMES = ("metadata_baseline", "classical_ml", "cnn")
CALIBRATION_SPLIT = "calibration"


def threshold_for_sensitivity(y_true, y_score, target: float) -> float:
    """Highest threshold t such that at least `target` of the positive cases
    have a score >= t. It is the score of the positive case ranked
    ceil(target * n_positive) from the top."""
    if not 0 < target <= 1:
        raise ValueError("target sensitivity must be in (0, 1]")
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    positives = np.sort(y_score[y_true == 1])[::-1]
    if len(positives) == 0:
        raise ValueError("no positive cases: a sensitivity target cannot be met")
    needed = int(np.ceil(target * len(positives) - 1e-9))
    return float(positives[needed - 1])


def operating_point(y_true, y_score, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    flagged = np.asarray(y_score, dtype=float) >= threshold
    positive = y_true == 1
    return {
        "n": int(len(y_true)),
        "n_positive": int(positive.sum()),
        "sensitivity": float((flagged & positive).sum() / positive.sum()),
        "specificity": float((~flagged & ~positive).sum() / (~positive).sum()),
        "flagged_fraction": float(flagged.mean()),
    }


def freeze_model_threshold(model: str, reports_dir: Path, target: float = 0.90, split: str = "tune") -> dict:
    """Raises FileNotFoundError if predictions or the calibrator are missing,
    ValueError if the split must not be used."""
    if split == HELD_OUT_SPLIT:
        raise ValueError("thresholds must not be chosen on the held-out test set")
    if split == CALIBRATION_SPLIT:
        raise ValueError("the calibration subset is reserved for fitting calibrators; choose thresholds on the tune subset")

    model_dir = Path(reports_dir) / model
    predictions_path = model_dir / f"predictions_{split}.csv"
    calibrator_path = model_dir / "calibrator.json"
    for path in (predictions_path, calibrator_path):
        if not path.exists():
            raise FileNotFoundError(path)

    predictions = pd.read_csv(predictions_path)
    other = set(predictions["split"].unique()) - {split}
    if other:
        raise ValueError(f"{predictions_path} contains rows from other splits: {sorted(other)}")
    calibrator = load_calibrator(calibrator_path)
    if calibrator["slope"] <= 0:
        raise ValueError(f"{model}: calibrator slope is not positive, so raw and recalibrated scores rank differently")

    y_true, raw = predictions["label"].to_numpy(), predictions["prob"].to_numpy()
    threshold_raw = threshold_for_sensitivity(y_true, raw, target)
    return {
        "model": model,
        "rule": f"highest threshold with sensitivity >= {target:.2f} on the {split} subset; flag when score >= threshold",
        "target_sensitivity": target,
        "threshold_raw": threshold_raw,
        "threshold_recalibrated": float(apply_recalibration([threshold_raw], calibrator)[0]),
        "decision_scale": "raw probability (threshold_raw); threshold_recalibrated is the same point for display",
        "selected_on": split,
        split: operating_point(y_true, raw, threshold_raw),
        "predictions_file": predictions_path.as_posix(),
        "calibrator_file": calibrator_path.as_posix(),
    }


def load_threshold(path: Path) -> dict:
    with open(path) as f:
        threshold = json.load(f)
    missing = {"threshold_raw", "threshold_recalibrated"} - set(threshold)
    if missing:
        raise ValueError(f"{path} is missing {missing}")
    return threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/v2"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--target-sensitivity", type=float, default=0.90)
    parser.add_argument("--split", default="tune")
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    args = parser.parse_args()

    frozen = 0
    for model in args.models:
        try:
            threshold = freeze_model_threshold(model, args.reports_dir, args.target_sensitivity, args.split)
        except FileNotFoundError as missing:
            print(f"{model:<20} skipped: {missing} not found")
            continue
        except ValueError as problem:
            sys.exit(f"Not frozen: {problem}")

        save_calibrator(threshold, args.models_dir / f"{model}_threshold.json")  # plain JSON writer
        save_calibrator(threshold, args.reports_dir / model / "threshold.json")
        frozen += 1
        point = threshold[args.split]
        print(
            f"{model:<20} threshold raw {threshold['threshold_raw']:.3f} / recalibrated "
            f"{threshold['threshold_recalibrated']:.3f}   on {args.split}: sensitivity {point['sensitivity']:.3f}, "
            f"specificity {point['specificity']:.3f}, flagged {point['flagged_fraction']:.2f}"
        )

    if frozen == 0:
        sys.exit("No thresholds were frozen.")


if __name__ == "__main__":
    main()
