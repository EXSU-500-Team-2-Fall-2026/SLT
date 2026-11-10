"""Model 2 of 3: classical ML on hand-crafted dermoscopic image features
(see src/features/handcrafted.py and README.md). Gradient-boosted trees
over ABCD-style shape/color/texture statistics -- no raw pixels, no deep
learning -- sitting between the metadata-only baseline and the CNN.

Trains on the images listed in <splits-dir>/train.csv and reports on one
development split (--eval-split, default val). The held-out test set is
refused here: it is scored once, by the final evaluation, after
everything is frozen.

Features are looked up by image_id in the cached tables under
--features-dir (built by src/features/build_feature_table.py). A feature
row depends only on its image, so the same cached tables serve any split
of the same images; nothing needs recomputing when the split changes.

Usage:
    # original split (midterm results)
    python -m src.models.classical_ml

    # duplicate-aware split: report on tune, save predictions for the calibrator
    python -m src.models.classical_ml \\
        --splits-dir data/splits_v2 --eval-split tune --predict-splits tune calibration \\
        --reports-dir reports/v2/classical_ml
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from src.eval.metrics import reliability_diagram
from src.eval.reporting import add_split_arguments, check_development_splits, evaluate, split_label, write_predictions
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


def load_feature_pool(features_dir: Path) -> pd.DataFrame:
    """All cached feature rows under features_dir, one per image_id."""
    tables = [load_features(path) for path in sorted(Path(features_dir).glob("*.csv"))]
    if not tables:
        raise FileNotFoundError(f"no feature tables (*.csv) in {features_dir}; run src/features/build_feature_table.py")
    pool = pd.concat(tables, ignore_index=True)
    return pool.drop_duplicates(subset="image_id", keep="first")[["image_id", *FEATURE_NAMES]]


def load_split_features(split_csv: Path, pool: pd.DataFrame) -> pd.DataFrame:
    """The split manifest (identifiers and label, in manifest order) with
    each image's cached features attached."""
    split_df = pd.read_csv(split_csv)
    df = split_df.merge(pool, on="image_id", how="left", validate="one_to_one", indicator=True)
    without = df.loc[df["_merge"] == "left_only", "image_id"]
    if len(without):
        raise ValueError(
            f"{len(without)} image(s) in {split_csv} have no cached features, e.g. {list(without[:3])}; "
            "run src/features/build_feature_table.py for this split"
        )
    return df.drop(columns="_merge")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--out", type=Path, default=Path("models/classical_ml.joblib"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/classical_ml"))
    parser.add_argument("--threshold", type=float, default=0.5)
    add_split_arguments(parser)
    args = parser.parse_args()
    check_development_splits(parser, args.eval_split, *args.predict_splits)

    pool = load_feature_pool(args.features_dir)
    train_df = load_split_features(args.splits_dir / "train.csv", pool)
    eval_df = load_split_features(args.splits_dir / f"{args.eval_split}.csv", pool)

    model = build_pipeline()
    model.fit(train_df[FEATURE_NAMES], train_df["label"])

    eval_prob = model.predict_proba(eval_df[FEATURE_NAMES])[:, 1]
    sections, calib = evaluate(eval_df["label"], eval_prob, threshold=args.threshold, n_boot=args.n_boot)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    reliability_diagram(
        calib, f"Classical ML (hand-crafted features) – {split_label(args.eval_split)}",
        args.reports_dir / "reliability_diagram.png",
    )

    for split in args.predict_splits:
        split_df = eval_df if split == args.eval_split else load_split_features(args.splits_dir / f"{split}.csv", pool)
        split_prob = model.predict_proba(split_df[FEATURE_NAMES])[:, 1]
        write_predictions(split_df, split_prob, split, args.reports_dir / f"predictions_{split}.csv")

    perm = permutation_importance(
        model, eval_df[FEATURE_NAMES], eval_df["label"],
        n_repeats=10, random_state=42, scoring="roc_auc",
    )
    importances = sorted(
        zip(FEATURE_NAMES, perm.importances_mean.tolist()), key=lambda kv: kv[1], reverse=True
    )

    report = {
        "model": "classical_ml",
        "features": FEATURE_NAMES,
        "n_train": len(train_df),
        "n_val": len(eval_df),
        "splits_dir": args.splits_dir.as_posix(),
        "eval_split": args.eval_split,
        **sections,
        "feature_importances": importances,
    }
    with open(args.reports_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
