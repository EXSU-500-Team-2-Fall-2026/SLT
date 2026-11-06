"""Audit the committed train/validation/test splits for leakage and data
errors (course spec section 5.2: splits declared before modelling, made at
the group level, test set touched once).

Checks that need only the committed manifests in data/splits/ always run.
Checks that need the raw download (metadata tables, image files) run when
those files exist and are reported as SKIP otherwise, so the audit can be
rerun by anyone who clones the repository.

Every check ends as one of:
    PASS  nothing wrong
    WARN  a known, documented issue (see src/data/known_issues.py)
    FAIL  an error that must be fixed before modelling
    SKIP  the input it needs is not available

The script prints a table, writes it to reports/dataset/split_audit.md, and
exits with status 1 if any check FAILs.

Usage:
    python -m src.data.validate_splits                   # manifests + raw files if present
    python -m src.data.validate_splits --hash-images     # also look for byte-identical images
"""
import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.data.known_issues import NEAR_DUPLICATE_PAIRS, WITHDRAWN_TEST_IMAGES
from src.data.labels import ALL_CLASSES, dx_to_binary

SPLITS = ("train", "val", "test")
REQUIRED_COLUMNS = ["image_id", "lesion_id", "dx", "label", "split"]
MISSING_MARKERS = {"", "na", "n/a", "nan", "null", "none"}


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


def _examples(items, n: int = 5) -> str:
    items = sorted(items)
    shown = ", ".join(str(i) for i in items[:n])
    return shown + (f" (+{len(items) - n} more)" if len(items) > n else "")


def _is_missing(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.casefold().isin(MISSING_MARKERS)


def load_splits(splits_dir: Path) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Read train/val/test manifests as text so blanks and 'NA' stay visible."""
    splits, missing_files = {}, []
    for name in SPLITS:
        path = splits_dir / f"{name}.csv"
        if not path.is_file():
            missing_files.append(str(path))
            continue
        splits[name] = pd.read_csv(path, dtype=str, keep_default_na=False)
    return splits, missing_files


def load_table(path: Path) -> pd.DataFrame:
    """Read a metadata or ground-truth table, sniffing tab- vs comma-separated."""
    df = pd.read_csv(path, sep=None, engine="python", dtype=str, keep_default_na=False)
    for column in ("image_id", "lesion_id", "dx"):
        if column in df:
            df[column] = df[column].str.strip()
    if "dx" in df:
        df["dx"] = df["dx"].str.lower()
    return df


# ---------------------------------------------------------------------------
# Checks on the committed manifests only
# ---------------------------------------------------------------------------


def check_columns(splits: dict[str, pd.DataFrame]) -> CheckResult:
    problems = []
    for name, df in splits.items():
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            problems.append(f"{name}.csv lacks {missing}")
        elif (df["split"].str.strip() != name).any():
            n_bad = int((df["split"].str.strip() != name).sum())
            problems.append(f"{name}.csv has {n_bad} row(s) whose split column is not '{name}'")
    if problems:
        return CheckResult("columns_and_split_labels", "FAIL", "; ".join(problems))
    return CheckResult("columns_and_split_labels", "PASS", "all manifests have the expected columns and split labels")


def check_missing_and_labels(splits: dict[str, pd.DataFrame]) -> CheckResult:
    problems = []
    for name, df in splits.items():
        for column in ("image_id", "lesion_id", "dx", "label"):
            n_missing = int(_is_missing(df[column]).sum())
            if n_missing:
                problems.append(f"{name}: {n_missing} missing {column}")
        dx = df["dx"].str.strip().str.lower()
        bad_dx = sorted(set(dx[~dx.isin(ALL_CLASSES) & ~_is_missing(df["dx"])]))
        if bad_dx:
            problems.append(f"{name}: unrecognised dx {bad_dx}")
            continue
        valid = dx.isin(ALL_CLASSES) & ~_is_missing(df["label"])
        expected = dx[valid].map(dx_to_binary).astype(str)
        n_wrong = int((df.loc[valid, "label"].str.strip() != expected).sum())
        if n_wrong:
            problems.append(f"{name}: {n_wrong} label(s) disagree with dx")
    if problems:
        return CheckResult("missing_values_and_labels", "FAIL", "; ".join(problems))
    return CheckResult(
        "missing_values_and_labels", "PASS",
        "no missing image_id/lesion_id/dx/label; every label matches its dx",
    )


def check_duplicates_within_splits(splits: dict[str, pd.DataFrame]) -> CheckResult:
    problems = []
    for name, df in splits.items():
        n_rows = int(df.duplicated().sum())
        dup_ids = set(df.loc[df["image_id"].duplicated(), "image_id"])
        if n_rows:
            problems.append(f"{name}: {n_rows} duplicate row(s)")
        if dup_ids:
            problems.append(f"{name}: image_id repeated ({_examples(dup_ids)})")
    if problems:
        return CheckResult("no_duplicates_within_split", "FAIL", "; ".join(problems))
    return CheckResult("no_duplicates_within_split", "PASS", "no repeated rows or image_ids within any split")


def _ids_in_more_than_one_split(splits: dict[str, pd.DataFrame], column: str) -> dict[str, set[str]]:
    seen: dict[str, set[str]] = {}
    for name, df in splits.items():
        for value in set(df[column].str.strip()):
            seen.setdefault(value, set()).add(name)
    return {value: names for value, names in seen.items() if len(names) > 1}


def check_image_ids_disjoint(splits: dict[str, pd.DataFrame]) -> CheckResult:
    shared = _ids_in_more_than_one_split(splits, "image_id")
    if shared:
        return CheckResult("image_ids_disjoint", "FAIL", f"{len(shared)} image_id(s) in more than one split: {_examples(shared)}")
    return CheckResult("image_ids_disjoint", "PASS", "every image_id appears in exactly one split")


def check_lesion_ids_disjoint(splits: dict[str, pd.DataFrame]) -> CheckResult:
    shared = _ids_in_more_than_one_split(splits, "lesion_id")
    if shared:
        return CheckResult("lesion_ids_disjoint", "FAIL", f"{len(shared)} lesion_id(s) in more than one split: {_examples(shared)}")
    counts = ", ".join(f"{name} {df['lesion_id'].nunique():,}" for name, df in splits.items())
    return CheckResult("lesion_ids_disjoint", "PASS", f"no lesion crosses a split boundary (lesions: {counts})")


def find_crossing_pairs(splits: dict[str, pd.DataFrame], pairs=NEAR_DUPLICATE_PAIRS) -> tuple[list, list]:
    """Return (pairs whose two images sit in different splits, pairs not found)."""
    where = {}
    for name, df in splits.items():
        for image_id in df["image_id"].str.strip():
            where[image_id] = name
    crossing, not_found = [], []
    for a, b in pairs:
        if a not in where or b not in where:
            not_found.append((a, b))
        elif where[a] != where[b]:
            crossing.append((a, where[a], b, where[b]))
    return crossing, not_found


def check_near_duplicate_pairs(splits: dict[str, pd.DataFrame]) -> CheckResult:
    crossing, not_found = find_crossing_pairs(splits)
    if not_found:
        return CheckResult(
            "known_near_duplicates_same_split", "FAIL",
            f"{len(not_found)} known pair(s) not found in the manifests: {_examples(not_found)}",
        )
    if crossing:
        listed = "; ".join(f"{a} ({sa}) / {b} ({sb})" for a, sa, b, sb in crossing)
        return CheckResult(
            "known_near_duplicates_same_split", "WARN",
            f"{len(crossing)} of {len(NEAR_DUPLICATE_PAIRS)} known near-duplicate pairs cross splits "
            f"(they have different lesion_ids, so a lesion-level split cannot see them): {listed}. "
            "Planned fix: merge each pair into one group in make_splits.py before splitting.",
        )
    return CheckResult(
        "known_near_duplicates_same_split", "PASS",
        f"all {len(NEAR_DUPLICATE_PAIRS)} known near-duplicate pairs sit within a single split",
    )


# ---------------------------------------------------------------------------
# Checks that need the raw download
# ---------------------------------------------------------------------------


def _compare_to_source(manifest: pd.DataFrame, source: pd.DataFrame, label: str) -> list[str]:
    """Problems when manifest rows disagree with the source table they came from."""
    problems = []
    merged = manifest.merge(source[["image_id", "lesion_id", "dx"]], on="image_id", how="left",
                            suffixes=("", "_source"), indicator=True)
    not_in_source = set(merged.loc[merged["_merge"] == "left_only", "image_id"])
    if not_in_source:
        problems.append(f"{len(not_in_source)} {label} image_id(s) not in the source table: {_examples(not_in_source)}")
    found = merged[merged["_merge"] == "both"]
    wrong_lesion = set(found.loc[found["lesion_id"] != found["lesion_id_source"], "image_id"])
    wrong_dx = set(found.loc[found["dx"].str.lower() != found["dx_source"], "image_id"])
    if wrong_lesion:
        problems.append(f"{len(wrong_lesion)} {label} image(s) with a different lesion_id: {_examples(wrong_lesion)}")
    if wrong_dx:
        problems.append(f"{len(wrong_dx)} {label} image(s) with a different dx: {_examples(wrong_dx)}")
    return problems


def check_train_val_against_metadata(splits: dict[str, pd.DataFrame], metadata_path: Path) -> CheckResult:
    name = "train_val_match_ham10000_metadata"
    if not metadata_path.is_file():
        return CheckResult(name, "SKIP", f"{metadata_path} not found")
    metadata = load_table(metadata_path)
    train_val = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    problems = _compare_to_source(train_val, metadata, "train/val")
    unassigned = set(metadata["image_id"]) - set(train_val["image_id"])
    if unassigned:
        problems.append(f"{len(unassigned)} metadata image(s) in neither train nor val: {_examples(unassigned)}")
    if problems:
        return CheckResult(name, "FAIL", "; ".join(problems))
    return CheckResult(
        name, "PASS",
        f"all {len(metadata):,} metadata images are in exactly one of train/val, with matching lesion_id and dx",
    )


def check_test_against_ground_truth(splits: dict[str, pd.DataFrame], ground_truth_path: Path) -> CheckResult:
    # HAM10000_metadata does not list the test images, so test.csv is checked
    # against the ISIC 2018 Task 3 ground-truth file it was built from.
    name = "test_matches_isic2018_ground_truth"
    if not ground_truth_path.is_file():
        return CheckResult(name, "SKIP", f"{ground_truth_path} not found")
    ground_truth = load_table(ground_truth_path)
    test = splits["test"]
    problems = _compare_to_source(test, ground_truth, "test")
    unlisted = set(ground_truth["image_id"]) - set(test["image_id"])
    if unlisted:
        problems.append(f"{len(unlisted)} ground-truth image(s) missing from test.csv: {_examples(unlisted)}")
    if problems:
        return CheckResult(name, "FAIL", "; ".join(problems))
    return CheckResult(name, "PASS", f"test.csv matches all {len(ground_truth):,} ground-truth rows")


def _unreadable_reason(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            image.verify()
    except (Image.DecompressionBombError, OSError, SyntaxError, UnidentifiedImageError, ValueError) as error:
        return str(error) or error.__class__.__name__
    return None


def check_images_readable(splits: dict[str, pd.DataFrame], image_dirs: dict[str, Path]) -> CheckResult:
    name = "image_files_exist_and_open"
    missing_dirs = sorted({str(d) for d in image_dirs.values() if not d.is_dir()})
    if missing_dirs:
        return CheckResult(name, "SKIP", f"image folder(s) not found: {', '.join(missing_dirs)}")
    missing, unreadable, withdrawn, checked = [], [], [], 0
    for split, df in splits.items():
        for image_id in df["image_id"].str.strip():
            path = image_dirs[split] / f"{image_id}.jpg"
            if not path.is_file():
                (withdrawn if split == "test" and image_id in WITHDRAWN_TEST_IMAGES else missing).append(image_id)
                continue
            checked += 1
            reason = _unreadable_reason(path)
            if reason:
                unreadable.append(f"{image_id} ({reason})")
    problems = []
    if missing:
        problems.append(f"{len(missing)} image file(s) missing: {_examples(missing)}")
    if unreadable:
        problems.append(f"{len(unreadable)} image file(s) unreadable: {_examples(unreadable)}")
    if problems:
        return CheckResult(name, "FAIL", "; ".join(problems))
    if withdrawn:
        return CheckResult(
            name, "WARN",
            f"{checked:,} images open; known withdrawn test image(s) absent as expected: {_examples(withdrawn)}",
        )
    return CheckResult(name, "PASS", f"all {checked:,} images exist and open")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_identical_image_files(splits: dict[str, pd.DataFrame], image_dirs: dict[str, Path]) -> CheckResult:
    """Byte-identical files under different image_ids, across all three splits."""
    name = "no_byte_identical_images"
    if any(not d.is_dir() for d in image_dirs.values()):
        return CheckResult(name, "SKIP", "image folder(s) not found")
    by_hash: dict[str, list[tuple[str, str, str]]] = {}
    for split, df in splits.items():
        for image_id, lesion_id in zip(df["image_id"].str.strip(), df["lesion_id"].str.strip(), strict=True):
            path = image_dirs[split] / f"{image_id}.jpg"
            if path.is_file():
                by_hash.setdefault(_sha256(path), []).append((image_id, lesion_id, split))
    groups = [g for g in by_hash.values() if len(g) > 1]
    across = [g for g in groups if len({s for _, _, s in g}) > 1]
    other_lesion = [g for g in groups if len({s for _, _, s in g}) == 1 and len({l for _, l, _ in g}) > 1]
    def describe(gs):
        return "; ".join(" = ".join(f"{i} ({les}, {s})" for i, les, s in g) for g in gs)

    if across:
        return CheckResult(name, "FAIL", f"{len(across)} identical-file group(s) span splits: {describe(across)}")
    if other_lesion:
        return CheckResult(
            name, "WARN",
            f"{len(other_lesion)} identical-file group(s) within one split under different lesion_ids: {describe(other_lesion)}",
        )
    same_lesion = len(groups)
    return CheckResult(name, "PASS", f"no identical files across splits ({same_lesion} identical group(s) share a lesion_id)")


# ---------------------------------------------------------------------------


def run_audit(splits_dir: Path, metadata_path: Path, ground_truth_path: Path,
              image_dirs: dict[str, Path], hash_images: bool) -> list[CheckResult]:
    splits, missing_files = load_splits(splits_dir)
    if missing_files:
        return [CheckResult("split_files_present", "FAIL", f"missing: {', '.join(missing_files)}")]
    results = [check_columns(splits)]
    if results[0].status == "FAIL":
        return results
    results += [
        check_missing_and_labels(splits),
        check_duplicates_within_splits(splits),
        check_image_ids_disjoint(splits),
        check_lesion_ids_disjoint(splits),
        check_near_duplicate_pairs(splits),
        check_train_val_against_metadata(splits, metadata_path),
        check_test_against_ground_truth(splits, ground_truth_path),
        check_images_readable(splits, image_dirs),
    ]
    if hash_images:
        results.append(check_identical_image_files(splits, image_dirs))
    else:
        results.append(CheckResult("no_byte_identical_images", "SKIP", "run with --hash-images"))
    return results


def to_markdown(results: list[CheckResult]) -> str:
    lines = [
        "# Split audit",
        "",
        "Generated by `python -m src.data.validate_splits`. Do not edit by hand.",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    lines += [f"| `{r.name}` | {r.status} | {r.detail.replace('|', '/')} |" for r in results]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--metadata", type=Path, default=None, help="default: <raw-dir>/HAM10000_metadata.tab")
    parser.add_argument("--test-ground-truth", type=Path, default=None,
                        help="default: <raw-dir>/ISIC2018_Task3_Test_GroundTruth.tab")
    parser.add_argument("--train-image-dir", type=Path, default=None, help="default: <raw-dir>/HAM10000_images")
    parser.add_argument("--test-image-dir", type=Path, default=None, help="default: <raw-dir>/ISIC2018_Task3_Test_Images")
    parser.add_argument("--hash-images", action="store_true", help="also hash every image to find byte-identical files")
    parser.add_argument("--out", type=Path, default=Path("reports/dataset/split_audit.md"))
    args = parser.parse_args()

    train_dir = args.train_image_dir or args.raw_dir / "HAM10000_images"
    test_dir = args.test_image_dir or args.raw_dir / "ISIC2018_Task3_Test_Images"
    results = run_audit(
        splits_dir=args.splits_dir,
        metadata_path=args.metadata or args.raw_dir / "HAM10000_metadata.tab",
        ground_truth_path=args.test_ground_truth or args.raw_dir / "ISIC2018_Task3_Test_GroundTruth.tab",
        image_dirs={"train": train_dir, "val": train_dir, "test": test_dir},
        hash_images=args.hash_images,
    )

    width = max(len(r.name) for r in results)
    for r in results:
        print(f"{r.status:4s}  {r.name:<{width}}  {r.detail}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(to_markdown(results))
    print(f"\nwrote {args.out}")
    return 1 if any(r.status == "FAIL" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
