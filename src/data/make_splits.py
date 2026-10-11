"""Build the train/validation/test splits for the HAM10000 binary
malignancy task, declared once before any model is trained.

- Train/validation come from HAM10000_metadata.tab, split at the LESION
  level (GroupShuffleSplit on lesion_id) so images of the same lesion never
  appear in more than one split.
- Test is the official ISIC 2018 Task 3 test set. The Harvard Dataverse
  record gives its ground truth as a tab file with the same
  lesion_id/image_id/dx schema as the training metadata (not the one-hot
  MEL/NV/.../ columns of the original ISIC challenge CSV), and it is
  touched only at final evaluation.

Usage:
    python -m src.data.make_splits \\
        --metadata data/raw/HAM10000_metadata.tab \\
        --test-ground-truth data/raw/ISIC2018_Task3_Test_GroundTruth.tab \\
        --out data/splits --val-size 0.15 --seed 42
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.data.labels import dx_to_binary

OUTPUT_COLUMNS = ["image_id", "lesion_id", "dx", "label", "split"]


def load_dx_table(path: Path) -> pd.DataFrame:
    """Load a lesion_id/image_id/dx table, sniffing tab- vs comma-separated."""
    df = pd.read_csv(path, sep=None, engine="python")
    required = {"lesion_id", "image_id", "dx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    df["dx"] = df["dx"].str.strip().str.lower()
    df["label"] = df["dx"].map(dx_to_binary)
    return df


def load_train_val_metadata(metadata_path: Path) -> pd.DataFrame:
    return load_dx_table(metadata_path)


def split_train_val(df: pd.DataFrame, val_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=df["lesion_id"]))
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    overlap = set(train_df["lesion_id"]) & set(val_df["lesion_id"])
    assert not overlap, f"Lesion-level leakage between train and val: {overlap}"

    train_df["split"] = "train"
    val_df["split"] = "val"
    return train_df, val_df


def load_test_ground_truth(path: Path) -> pd.DataFrame:
    df = load_dx_table(path)
    df["split"] = "test"
    return df[OUTPUT_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--test-ground-truth", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/splits"))
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    df = load_train_val_metadata(args.metadata)
    train_df, val_df = split_train_val(df, args.val_size, args.seed)

    train_df[OUTPUT_COLUMNS].to_csv(args.out / "train.csv", index=False)
    val_df[OUTPUT_COLUMNS].to_csv(args.out / "val.csv", index=False)
    print(f"train: {len(train_df):5d} images, {train_df['lesion_id'].nunique():5d} lesions")
    print(f"val:   {len(val_df):5d} images, {val_df['lesion_id'].nunique():5d} lesions")

    if args.test_ground_truth is not None:
        test_df = load_test_ground_truth(args.test_ground_truth)
        test_df.to_csv(args.out / "test.csv", index=False)
        print(f"test:  {len(test_df):5d} images (official ISIC 2018 Task 3 test set)")
    else:
        print("No --test-ground-truth given; test.csv not written.")


if __name__ == "__main__":
    main()
