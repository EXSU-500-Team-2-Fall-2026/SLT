"""Extract hand-crafted features (src/features/handcrafted.py) for every
image in one or more splits and cache them to CSV, so the classical-ML
model doesn't need to re-run feature extraction (and so a teammate without
the raw images, but with these committed CSVs, can still load the
features). Mirrors the train/val-only convention of model 1: the test
split is extracted separately, once, at final model comparison time.

Usage:
    python -m src.features.build_feature_table \\
        --splits-dir data/splits --raw-dir data/raw --out data/features \\
        --splits train val
"""
import argparse
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.features.handcrafted import FEATURE_NAMES, extract_features_from_path

IMAGE_DIR_BY_SPLIT = {
    "train": "HAM10000_images",
    "val": "HAM10000_images",
    "test": "ISIC2018_Task3_Test_Images",
}


def build_table(split_csv: Path, image_dir: Path) -> pd.DataFrame:
    split_df = pd.read_csv(split_csv)
    rows = []
    missing = []
    for row in tqdm(split_df.itertuples(index=False), total=len(split_df), desc=split_csv.stem):
        image_path = image_dir / f"{row.image_id}.jpg"
        if not image_path.exists():
            missing.append(row.image_id)
            continue
        features = extract_features_from_path(image_path)
        rows.append({"image_id": row.image_id, "label": row.label, **features})

    if missing:
        print(f"  skipped {len(missing)} image(s) with no file on disk: {missing[:5]}...")

    return pd.DataFrame(rows, columns=["image_id", "label", *FEATURE_NAMES])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/features"))
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")
    args.out.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        split_csv = args.splits_dir / f"{split}.csv"
        image_dir = args.raw_dir / IMAGE_DIR_BY_SPLIT[split]
        df = build_table(split_csv, image_dir)
        out_path = args.out / f"{split}.csv"
        df.to_csv(out_path, index=False)
        print(f"{split}: wrote {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
