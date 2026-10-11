"""Model 1 of 3: patient-metadata-only baseline for the malignancy triage
task (see README.md). Uses only age, sex, and body site (localization) --
no image data -- via a logistic regression on an impute+encode pipeline.
Establishes the floor the image-based models (classical ML, CNN) must beat.

Trains on data/splits/train.csv and reports on data/splits/val.csv only;
the held-out test set is touched once, after all three models are built
and compared.

Usage:
    python -m src.models.metadata_baseline \\
        --splits-dir data/splits --raw-dir data/raw \\
        --out models/metadata_baseline.joblib \\
        --reports-dir reports/metadata_baseline
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.eval.metrics import bootstrap_metrics_ci, calibration_metrics, discrimination_metrics, reliability_diagram

NUMERIC_FEATURES = ["age"]
CATEGORICAL_FEATURES = ["sex", "localization"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_pipeline() -> Pipeline:
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )
    return Pipeline(
        [("preprocess", preprocess), ("classify", LogisticRegression(max_iter=1000, class_weight="balanced"))]
    )


def load_split_with_metadata(split_csv: Path, metadata_path: Path) -> pd.DataFrame:
    """Join a split manifest (image_id, label, ...) with the raw metadata
    table to pull in age/sex/localization, which make_splits.py doesn't
    carry through."""
    split_df = pd.read_csv(split_csv)
    meta_df = pd.read_csv(metadata_path, sep=None, engine="python")
    meta_df = meta_df[["image_id", *CATEGORICAL_FEATURES, *NUMERIC_FEATURES]]
    df = split_df.merge(meta_df, on="image_id", how="left", validate="one_to_one")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("models/metadata_baseline.joblib"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/metadata_baseline"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    train_metadata = args.raw_dir / "HAM10000_metadata.tab"
    train_df = load_split_with_metadata(args.splits_dir / "train.csv", train_metadata)
    val_df = load_split_with_metadata(args.splits_dir / "val.csv", train_metadata)

    pipeline = build_pipeline()
    pipeline.fit(train_df[FEATURE_COLUMNS], train_df["label"])

    val_prob = pipeline.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]
    disc = discrimination_metrics(val_df["label"], val_prob, threshold=args.threshold)
    calib = calibration_metrics(val_df["label"], val_prob)
    ci = bootstrap_metrics_ci(val_df["label"], val_prob, threshold=args.threshold)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.out)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    reliability_diagram(
        calib, "Metadata baseline – validation", args.reports_dir / "reliability_diagram.png"
    )

    report = {
        "model": "metadata_baseline",
        "features": FEATURE_COLUMNS,
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
    }
    with open(args.reports_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
