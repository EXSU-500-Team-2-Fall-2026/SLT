"""Duplicate-aware splits: unit tests on small synthetic tables, plus a check
that the committed original manifests regroup cleanly."""
from pathlib import Path

import pandas as pd
import pytest

from src.data import make_splits_v2 as v2
from src.data.known_issues import NEAR_DUPLICATE_PAIRS

SPLITS_DIR = Path(__file__).resolve().parents[1] / "data" / "splits"


def _table(n_lesions=200, images_per_lesion=2):
    rows = []
    for i in range(n_lesions):
        for j in range(images_per_lesion):
            rows.append(
                {"image_id": f"IMG_{i:04d}_{j}", "lesion_id": f"LES_{i:04d}", "dx": "mel" if i % 5 == 0 else "nv",
                 "label": int(i % 5 == 0)}
            )
    return pd.DataFrame(rows)


def test_duplicate_pair_merges_two_lesions():
    df = _table(4)
    groups = v2.assign_group_ids(df, [("IMG_0000_0", "IMG_0003_1")])
    assert groups[df["lesion_id"] == "LES_0000"].unique().tolist() == ["LES_0000"]
    assert groups[df["lesion_id"] == "LES_0003"].unique().tolist() == ["LES_0000"]
    assert groups.nunique() == 3


def test_links_are_transitive():
    df = _table(5)
    pairs = [("IMG_0001_0", "IMG_0002_0"), ("IMG_0002_1", "IMG_0004_0")]
    groups = v2.assign_group_ids(df, pairs)
    linked = groups[df["lesion_id"].isin(["LES_0001", "LES_0002", "LES_0004"])]
    assert linked.nunique() == 1
    assert groups.nunique() == 3


def test_pairs_with_unknown_images_are_ignored_and_reported():
    df = _table(3)
    pairs = [("IMG_0000_0", "NOT_THERE")]
    assert v2.assign_group_ids(df, pairs).nunique() == 3
    assert v2.find_unmatched_pairs(df, pairs) == pairs


def test_split_keeps_groups_together_and_is_reproducible():
    df = _table(400)
    first = v2.split_development(df, seed=42)
    second = v2.split_development(df.sample(frac=1, random_state=1), seed=42)

    assert set(first["split"]) == set(v2.DEV_SPLITS)
    assert first.groupby("group_id")["split"].nunique().max() == 1
    share = first.groupby("split")["group_id"].nunique() / first["group_id"].nunique()
    assert share["train"] == pytest.approx(0.85, abs=0.01)
    assert share["tune"] == pytest.approx(0.075, abs=0.01)
    assert share["calibration"] == pytest.approx(0.075, abs=0.01)

    # same assignment regardless of row order
    a = first.set_index("image_id")["split"].sort_index()
    b = second.set_index("image_id")["split"].sort_index()
    assert a.equals(b)


def test_check_disjoint_catches_a_crossing_group():
    df = v2.split_development(_table(100))
    victim = df.index[df["split"] == "train"][0]
    df.loc[victim, "split"] = "tune"
    with pytest.raises(AssertionError):
        v2.check_disjoint(df)


def test_invalid_sizes_rejected():
    with pytest.raises(ValueError):
        v2.split_development(_table(50), tune_size=0.6, calibration_size=0.5)


def test_committed_manifests_regroup_without_crossing_duplicates():
    dev = v2.load_development_table(None, SPLITS_DIR)
    assert not v2.find_unmatched_pairs(dev), "a known near-duplicate image is missing from the manifests"

    out = v2.split_development(dev)
    split_of = dict(zip(out["image_id"], out["split"]))
    crossing = [(a, b) for a, b in NEAR_DUPLICATE_PAIRS if split_of[a] != split_of[b]]
    assert crossing == []
    assert len(out) == len(dev)
