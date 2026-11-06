"""Split integrity: checks on the committed manifests in data/splits/ (no raw
data needed) and unit tests of src/data/validate_splits.py on small
synthetic splits."""
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.data import validate_splits as vs

SPLITS_DIR = Path(__file__).resolve().parents[1] / "data" / "splits"


@pytest.fixture(scope="module")
def committed():
    splits, missing = vs.load_splits(SPLITS_DIR)
    assert not missing, f"missing split files: {missing}"
    return splits


@pytest.mark.parametrize(
    "check",
    [
        vs.check_columns,
        vs.check_missing_and_labels,
        vs.check_duplicates_within_splits,
        vs.check_image_ids_disjoint,
        vs.check_lesion_ids_disjoint,
    ],
)
def test_committed_splits_pass(committed, check):
    result = check(committed)
    assert result.status == "PASS", result.detail


@pytest.mark.xfail(
    strict=True,
    reason="5 of the 18 known near-duplicate pairs cross train/val in the current split; "
    "fixed by merging the pairs in make_splits.py and re-splitting. Remove this marker then.",
)
def test_committed_splits_keep_near_duplicates_together(committed):
    crossing, not_found = vs.find_crossing_pairs(committed)
    assert not not_found
    assert not crossing


# --- synthetic splits ------------------------------------------------------


def _split(name, rows):
    df = pd.DataFrame(rows, columns=["image_id", "lesion_id", "dx"])
    df["label"] = df["dx"].map({"mel": "1", "nv": "0"})
    df["split"] = name
    return df.astype(str)


@pytest.fixture
def clean():
    return {
        "train": _split("train", [("IMG_1", "L1", "mel"), ("IMG_2", "L1", "mel"), ("IMG_3", "L2", "nv")]),
        "val": _split("val", [("IMG_4", "L3", "nv")]),
        "test": _split("test", [("IMG_5", "T1", "nv")]),
    }


def test_clean_splits_pass(clean):
    for check in (vs.check_missing_and_labels, vs.check_image_ids_disjoint, vs.check_lesion_ids_disjoint):
        assert check(clean).status == "PASS"


def test_lesion_in_two_splits_fails(clean):
    clean["val"] = _split("val", [("IMG_4", "L1", "mel")])
    result = vs.check_lesion_ids_disjoint(clean)
    assert result.status == "FAIL" and "L1" in result.detail


def test_label_that_disagrees_with_dx_fails(clean):
    clean["train"].loc[0, "label"] = "0"
    assert vs.check_missing_and_labels(clean).status == "FAIL"


def test_missing_marker_fails(clean):
    clean["val"].loc[0, "lesion_id"] = "NA"
    assert vs.check_missing_and_labels(clean).status == "FAIL"


def test_crossing_pair_is_reported(clean):
    crossing, not_found = vs.find_crossing_pairs(clean, pairs=[("IMG_1", "IMG_4"), ("IMG_1", "IMG_3")])
    assert crossing == [("IMG_1", "train", "IMG_4", "val")]
    assert not not_found


def test_test_set_checked_against_its_own_ground_truth(clean, tmp_path):
    ground_truth = tmp_path / "gt.tab"
    pd.DataFrame({"image_id": ["IMG_5", "IMG_6"], "lesion_id": ["T1", "T2"], "dx": ["nv", "mel"]}).to_csv(
        ground_truth, sep="\t", index=False
    )
    result = vs.check_test_against_ground_truth(clean, ground_truth)
    assert result.status == "FAIL" and "IMG_6" in result.detail


def test_identical_files_across_splits_fail(clean, tmp_path):
    for split, df in clean.items():
        for i, image_id in enumerate(df["image_id"]):
            Image.new("RGB", (4, 4), color=(i * 40, len(split) * 30, 7)).save(tmp_path / f"{image_id}.jpg")
    (tmp_path / "IMG_4.jpg").write_bytes((tmp_path / "IMG_3.jpg").read_bytes())  # train copy lands in val
    dirs = {split: tmp_path for split in clean}
    assert vs.check_identical_image_files(clean, dirs).status == "FAIL"
    assert vs.check_images_readable(clean, dirs).status == "PASS"
