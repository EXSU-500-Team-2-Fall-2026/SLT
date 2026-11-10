"""Model 1 of 3: patient-metadata-only baseline for the malignancy triage
task (see README.md). Uses only age, sex, and body site (localization) --
no image data -- via a logistic regression on an impute+encode pipeline.
Establishes the floor the image-based models (classical ML, CNN) must beat.

Trains on <splits-dir>/train.csv and reports on one development split
(--eval-split, default val). The held-out test set is refused here: it is
scored once, by the final evaluation, after everything is frozen.

Usage:
    # original split (midterm results)
    python -m src.models.metadata_baseline

    # duplicate-aware split: report on tune, save predictions for the calibrator
    python -m src.models.metadata_baseline \\
        --splits-dir data/splits_v2 --eval-split tune --predict-splits tune calibration \\
        --reports-dir reports/v2/metadata_baseline
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

from src.eval.metrics import reliability_diagram
from src.eval.reporting import add_split_arguments, check_development_splits, evaluate, split_label, write_predictions

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
    add_split_arguments(parser)
    args = parser.parse_args()
    check_development_splits(parser, args.eval_split, *args.predict_splits)

    metadata = args.raw_dir / "HAM10000_metadata.tab"
    train_df = load_split_with_metadata(args.splits_dir / "train.csv", metadata)
    eval_df = load_split_with_metadata(args.splits_dir / f"{args.eval_split}.csv", metadata)

    pipeline = build_pipeline()
    pipeline.fit(train_df[FEATURE_COLUMNS], train_df["label"])

    eval_prob = pipeline.predict_proba(eval_df[FEATURE_COLUMNS])[:, 1]
    sections, calib = evaluate(eval_df["label"], eval_prob, threshold=args.threshold, n_boot=args.n_boot)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.out)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    reliability_diagram(
        calib, f"Metadata baseline – {split_label(args.eval_split)}", args.reports_dir / "reliability_diagram.png"
    )

    for split in args.predict_splits:
        split_df = eval_df if split == args.eval_split else load_split_with_metadata(
            args.splits_dir / f"{split}.csv", metadata
        )
        split_prob = pipeline.predict_proba(split_df[FEATURE_COLUMNS])[:, 1]
        write_predictions(split_df, split_prob, split, args.reports_dir / f"predictions_{split}.csv")

    report = {
        "model": "metadata_baseline",
        "features": FEATURE_COLUMNS,
        "n_train": len(train_df),
        "n_val": len(eval_df),
        "splits_dir": args.splits_dir.as_posix(),
        "eval_split": args.eval_split,
        **sections,
    }
    with open(args.reports_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
