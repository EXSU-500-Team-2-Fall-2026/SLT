"""Duplicate-aware development splits (version 2).

Implements steps 2-4 of "Planned correction after midterm submission" in
reports/experimental-design.md. The original manifests in data/splits/ are
left untouched; this script writes new ones to data/splits_v2/.

What changes compared with src/data/make_splits.py:

- Grouping. Images are grouped by *connected group*: two images are in the
  same group if they share a lesion_id or are one of the known
  near-duplicate pairs (src/data/known_issues.py). Links are transitive.
  The original split grouped by lesion_id only, which let 5 of the 18
  near-duplicate pairs cross the train/validation boundary.
- Three development subsets instead of two: train (fit models), tune
  (model selection and CNN checkpoint selection) and calibration (fit the
  recalibration mapping only). Starting allocation 85 / 7.5 / 7.5 % of
  groups, seed 42.
- The held-out test manifest is copied unchanged, with group_id set to its
  lesion_id. The known near-duplicate pairs are all HAM10000 images, so
  they do not link development and test.

The allocation is by group, not by image and not stratified, so image
counts and class proportions will not be exactly 85 / 7.5 / 7.5. Check the
printed summary before freezing. Never pick a seed or allocation by
looking at model performance.

Usage (either source gives the same assignment, because groups are sorted
before shuffling):
    python -m src.data.make_splits_v2 --from-splits data/splits --out data/splits_v2
    python -m src.data.make_splits_v2 --metadata data/raw/HAM10000_metadata.tab --out data/splits_v2
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.data.known_issues import NEAR_DUPLICATE_PAIRS
from src.data.make_splits import load_dx_table

OUTPUT_COLUMNS = ["image_id", "lesion_id", "group_id", "dx", "label", "split"]
DEV_SPLITS = ("train", "tune", "calibration")


def assign_group_ids(df: pd.DataFrame, duplicate_pairs=NEAR_DUPLICATE_PAIRS) -> pd.Series:
    """Return one group_id per row: the smallest lesion_id in the connected
    component formed by shared lesion_ids and the given image pairs.

    Pairs that mention an image_id missing from df are ignored (reported by
    find_unmatched_pairs), so the function also works on subsets."""
    parent = {lesion: lesion for lesion in df["lesion_id"].unique()}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    lesion_of = dict(zip(df["image_id"], df["lesion_id"]))
    for a, b in duplicate_pairs:
        if a not in lesion_of or b not in lesion_of:
            continue
        root_a, root_b = find(lesion_of[a]), find(lesion_of[b])
        if root_a != root_b:
            keep, drop = sorted((root_a, root_b))
            parent[drop] = keep

    return df["lesion_id"].map(find).rename("group_id")


def find_unmatched_pairs(df: pd.DataFrame, duplicate_pairs=NEAR_DUPLICATE_PAIRS) -> list[tuple[str, str]]:
    known = set(df["image_id"])
    return [(a, b) for a, b in duplicate_pairs if a not in known or b not in known]


def split_development(
    df: pd.DataFrame, tune_size: float = 0.075, calibration_size: float = 0.075, seed: int = 42
) -> pd.DataFrame:
    """Add group_id and split columns. Whole groups go to one subset."""
    if tune_size <= 0 or calibration_size <= 0 or tune_size + calibration_size >= 1:
        raise ValueError("tune_size and calibration_size must be positive and sum to less than 1")

    df = df.copy()
    df["group_id"] = assign_group_ids(df)

    holdout = tune_size + calibration_size
    first = GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=seed)
    train_idx, held_idx = next(first.split(df, groups=df["group_id"]))

    held = df.iloc[held_idx]
    second = GroupShuffleSplit(n_splits=1, test_size=calibration_size / holdout, random_state=seed)
    tune_rel, cal_rel = next(second.split(held, groups=held["group_id"]))

    df["split"] = ""
    df.iloc[train_idx, df.columns.get_loc("split")] = "train"
    df.iloc[held_idx[tune_rel], df.columns.get_loc("split")] = "tune"
    df.iloc[held_idx[cal_rel], df.columns.get_loc("split")] = "calibration"

    check_disjoint(df)
    return df


def check_disjoint(df: pd.DataFrame) -> None:
    """Raise if any image, lesion or connected group is in more than one subset."""
    for column in ("image_id", "lesion_id", "group_id"):
        spread = df.groupby(column)["split"].nunique()
        crossing = spread[spread > 1]
        if len(crossing):
            raise AssertionError(f"{len(crossing)} {column} value(s) cross subsets, e.g. {list(crossing.index[:3])}")
    if (df["split"] == "").any():
        raise AssertionError("some rows were not assigned to a subset")


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, part in df.groupby("split", sort=False):
        rows.append(
            {
                "split": name,
                "images": len(part),
                "lesions": part["lesion_id"].nunique(),
                "groups": part["group_id"].nunique(),
                "positive_images": int(part["label"].sum()),
                "positive_pct": round(100 * part["label"].mean(), 2),
            }
        )
    order = {name: i for i, name in enumerate((*DEV_SPLITS, "test"))}
    return pd.DataFrame(rows).sort_values("split", key=lambda s: s.map(order)).reset_index(drop=True)


def load_development_table(metadata: Path | None, from_splits: Path | None) -> pd.DataFrame:
    if (metadata is None) == (from_splits is None):
        raise ValueError("give exactly one of --metadata or --from-splits")
    if metadata is not None:
        return load_dx_table(metadata)[["image_id", "lesion_id", "dx", "label"]]
    parts = [pd.read_csv(from_splits / f"{name}.csv") for name in ("train", "val")]
    return pd.concat(parts, ignore_index=True)[["image_id", "lesion_id", "dx", "label"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=Path, default=None, help="data/raw/HAM10000_metadata.tab")
    parser.add_argument("--from-splits", type=Path, default=None, help="existing data/splits directory")
    parser.add_argument("--test-manifest", type=Path, default=Path("data/splits/test.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/splits_v2"))
    parser.add_argument("--tune-size", type=float, default=0.075)
    parser.add_argument("--calibration-size", type=float, default=0.075)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dev = load_development_table(args.metadata, args.from_splits)
    if dev["image_id"].duplicated().any():
        raise ValueError("duplicate image_id rows in the development table")

    unmatched = find_unmatched_pairs(dev)
    if unmatched:
        print(f"WARNING: {len(unmatched)} near-duplicate pair(s) mention images not in the table: {unmatched}")

    dev = split_development(dev, args.tune_size, args.calibration_size, args.seed)
    merged = dev["lesion_id"].nunique() - dev["group_id"].nunique()
    print(f"{dev['lesion_id'].nunique()} lesions -> {dev['group_id'].nunique()} connected groups "
          f"({merged} lesion(s) merged by near-duplicate links)")

    args.out.mkdir(parents=True, exist_ok=True)
    for name in DEV_SPLITS:
        dev[dev["split"] == name][OUTPUT_COLUMNS].to_csv(args.out / f"{name}.csv", index=False)

    summary_input = dev
    if args.test_manifest.exists():
        test = pd.read_csv(args.test_manifest)
        test["group_id"] = test["lesion_id"]
        test["split"] = "test"
        overlap = set(test["image_id"]) & set(dev["image_id"])
        if overlap:
            raise AssertionError(f"{len(overlap)} image_id(s) are in both development and test")
        test[OUTPUT_COLUMNS].to_csv(args.out / "test.csv", index=False)
        summary_input = pd.concat([dev, test], ignore_index=True)
    else:
        print(f"No test manifest at {args.test_manifest}; test.csv not written.")

    summary = summarise(summary_input)
    summary.to_csv(args.out / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
