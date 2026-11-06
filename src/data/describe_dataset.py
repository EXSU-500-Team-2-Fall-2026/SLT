"""Describe the dataset and each split quantitatively, for the manuscript's
Data section and Table I (course spec section 6: size, classes,
distributions, known limitations).

The models score images, but several images can show the same lesion, so
every count is given both ways: images and lesions. Age, sex and body site
belong to the lesion and are counted once per lesion (HAM10000 has no
patient identifier). A lesion recorded at more than one body site is counted
at each of them. Test images withdrawn from the distribution
(src/data/known_issues.py) are excluded.

Writes three summary tables to reports/dataset/. They contain counts only,
never per-image rows, so no dataset content is redistributed:
    class_counts.csv   images and lesions per diagnosis, per split (Table I)
    split_summary.csv  size, malignant share, age, sex, missingness per split
    body_site.csv      lesions per body site, per split

class_counts.csv needs only the committed manifests; the other two need the
raw metadata (python -m src.data.download).

Usage:
    python -m src.data.describe_dataset
"""
import argparse
from pathlib import Path

import pandas as pd

from src.data.known_issues import WITHDRAWN_TEST_IMAGES
from src.data.labels import ALL_CLASSES

SPLITS = ("train", "val", "test")
GROUPS = ("ham10000", "train", "val", "test")  # ham10000 = train + val
MISSING_SEX = {"", "unknown", "nan"}
MISSING_SITE = {"", "unknown", "nan"}


def load_groups(splits_dir: Path) -> dict[str, pd.DataFrame]:
    """Split manifests keyed by group, with withdrawn test images removed."""
    frames = {name: pd.read_csv(splits_dir / f"{name}.csv") for name in SPLITS}
    frames["test"] = frames["test"][~frames["test"]["image_id"].isin(WITHDRAWN_TEST_IMAGES)]
    frames["ham10000"] = pd.concat([frames["train"], frames["val"]], ignore_index=True)
    return {group: frames[group].reset_index(drop=True) for group in GROUPS}


def class_counts(groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for group, df in groups.items():
        for dx in sorted(ALL_CLASSES):
            subset = df[df["dx"] == dx]
            rows.append({
                "group": group,
                "dx": dx,
                "label": int(subset["label"].iloc[0]) if len(subset) else None,
                "images": len(subset),
                "lesions": subset["lesion_id"].nunique(),
            })
    return pd.DataFrame(rows)


def attach_metadata(df: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Add age, sex and localization from the source table, matched by image_id."""
    columns = [c for c in ("age", "sex", "localization") if c in source.columns]
    extra = source[["image_id", *columns]].drop_duplicates("image_id")
    return df.merge(extra, on="image_id", how="left", validate="one_to_one")


def _normalised(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def lesion_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per lesion. Flags lesions whose images disagree on age or sex."""
    missing = pd.Series(float("nan"), index=df.index)
    df = df.assign(
        age=pd.to_numeric(df.get("age", missing), errors="coerce"),
        sex=_normalised(df["sex"]) if "sex" in df else "",
        localization=_normalised(df["localization"]) if "localization" in df else "",
    )
    grouped = df.groupby("lesion_id")
    lesions = pd.DataFrame({
        "label": grouped["label"].max(),
        "age": grouped["age"].first(),
        "age_values": grouped["age"].nunique(),
        "sex": grouped["sex"].first(),
        "sex_values": grouped["sex"].nunique(),
        "sites": grouped["localization"].agg(lambda s: sorted(set(s))),
    })
    return lesions


def split_summary(group: str, df: pd.DataFrame) -> dict:
    lesions = lesion_table(df)
    n_images, n_lesions = len(df), len(lesions)
    age = lesions["age"]
    sex = lesions["sex"]
    unknown_site = lesions["sites"].map(lambda sites: all(s in MISSING_SITE for s in sites))

    def pct(n: int, total: int) -> float:
        return round(100 * n / total, 2) if total else float("nan")

    return {
        "group": group,
        "images": n_images,
        "lesions": n_lesions,
        "malignant_images": int(df["label"].sum()),
        "malignant_images_pct": pct(int(df["label"].sum()), n_images),
        "malignant_lesions": int(lesions["label"].sum()),
        "malignant_lesions_pct": pct(int(lesions["label"].sum()), n_lesions),
        "age_median": age.median(),
        "age_q1": age.quantile(0.25),
        "age_q3": age.quantile(0.75),
        "age_mean": round(age.mean(), 2),
        "age_sd": round(age.std(), 2),
        "lesions_missing_age": int(age.isna().sum()),
        "male_lesions": int((sex == "male").sum()),
        "male_pct": pct(int((sex == "male").sum()), n_lesions),
        "female_lesions": int((sex == "female").sum()),
        "female_pct": pct(int((sex == "female").sum()), n_lesions),
        "unknown_sex_lesions": int(sex.isin(MISSING_SEX).sum()),
        "unknown_site_lesions": int(unknown_site.sum()),
        "lesions_with_conflicting_age": int((lesions["age_values"] > 1).sum()),
        "lesions_with_conflicting_sex": int((lesions["sex_values"] > 1).sum()),
        "lesions_with_several_sites": int(lesions["sites"].map(len).gt(1).sum()),
    }


def body_site_counts(group: str, df: pd.DataFrame) -> pd.DataFrame:
    lesions = lesion_table(df)
    exploded = lesions["sites"].explode().replace({s: "unknown" for s in MISSING_SITE})
    counts = exploded.value_counts().rename_axis("site").reset_index(name="lesions")
    counts["pct_of_lesions"] = (100 * counts["lesions"] / len(lesions)).round(2)
    counts.insert(0, "group", group)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--metadata", type=Path, default=None, help="default: <raw-dir>/HAM10000_metadata.tab")
    parser.add_argument("--test-ground-truth", type=Path, default=None,
                        help="default: <raw-dir>/ISIC2018_Task3_Test_GroundTruth.tab")
    parser.add_argument("--out", type=Path, default=Path("reports/dataset"))
    args = parser.parse_args()

    groups = load_groups(args.splits_dir)
    args.out.mkdir(parents=True, exist_ok=True)

    counts = class_counts(groups)
    counts.to_csv(args.out / "class_counts.csv", index=False)
    print(f"wrote {args.out / 'class_counts.csv'}")

    metadata_path = args.metadata or args.raw_dir / "HAM10000_metadata.tab"
    ground_truth_path = args.test_ground_truth or args.raw_dir / "ISIC2018_Task3_Test_GroundTruth.tab"
    if not metadata_path.is_file():
        print(f"{metadata_path} not found: skipping split_summary.csv and body_site.csv")
        return

    def read_table(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, sep=None, engine="python")

    sources = {"ham10000": read_table(metadata_path)}
    sources["train"] = sources["val"] = sources["ham10000"]
    if ground_truth_path.is_file():
        sources["test"] = read_table(ground_truth_path)
    else:
        print(f"{ground_truth_path} not found: test set left out of the demographic tables")

    summaries, sites = [], []
    for group in GROUPS:
        if group not in sources:
            continue
        df = attach_metadata(groups[group], sources[group])
        summaries.append(split_summary(group, df))
        sites.append(body_site_counts(group, df))

    pd.DataFrame(summaries).to_csv(args.out / "split_summary.csv", index=False)
    pd.concat(sites, ignore_index=True).to_csv(args.out / "body_site.csv", index=False)
    print(f"wrote {args.out / 'split_summary.csv'} and {args.out / 'body_site.csv'}")
    print(pd.DataFrame(summaries)[["group", "images", "lesions", "malignant_images_pct",
                                   "malignant_lesions_pct", "age_median", "male_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
