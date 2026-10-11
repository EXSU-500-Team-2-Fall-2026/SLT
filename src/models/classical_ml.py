"""Model 2 of 3: classical ML on hand-crafted dermoscopic image features
(see src/features/handcrafted.py and README.md). Gradient-boosted trees
over ABCD-style shape/color/texture statistics -- no raw pixels, no deep
learning -- sitting between the metadata-only baseline and the CNN.

Trains on data/features/train.csv and reports on data/features/val.csv
only; the held-out test set is touched once, after all three models are
built and compared. Run src/features/build_feature_table.py first to
produce those CSVs.

Usage:
    python -m src.models.classical_ml \\
        --features-dir data/features \\
        --out models/classical_ml.joblib \\
        --reports-dir reports/classical_ml
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from src.eval.metrics import bootstrap_metrics_ci, calibration_metrics, discrimination_metrics, reliability_diagram
from src.features.handcrafted import FEATURE_NAMES


def build_pipeline() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=300,
        class_weight="balanced",
        random_state=42,
    )


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(["image_id", "label", *FEATURE_NAMES]) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--out", type=Path, default=Path("models/classical_ml.joblib"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/classical_ml"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    train_df = load_features(args.features_dir / "train.csv")
    val_df = load_features(args.features_dir / "val.csv")

    model = build_pipeline()
    model.fit(train_df[FEATURE_NAMES], train_df["label"])

    val_prob = model.predict_proba(val_df[FEATURE_NAMES])[:, 1]
    disc = discrimination_metrics(val_df["label"], val_prob, threshold=args.threshold)
    calib = calibration_metrics(val_df["label"], val_prob)
    ci = bootstrap_metrics_ci(val_df["label"], val_prob, threshold=args.threshold)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    reliability_diagram(
        calib, "Classical ML (hand-crafted features) – validation",
        args.reports_dir / "reliability_diagram.png",
    )

    perm = permutation_importance(
        model, val_df[FEATURE_NAMES], val_df["label"],
        n_repeats=10, random_state=42, scoring="roc_auc",
    )
    importances = sorted(
        zip(FEATURE_NAMES, perm.importances_mean.tolist()), key=lambda kv: kv[1], reverse=True
    )

    report = {
        "model": "classical_ml",
        "features": FEATURE_NAMES,
        "n_train": len(train_df),
        "n_val": len(val_df),
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
        "feature_importances": importances,
    }
    with open(args.reports_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
