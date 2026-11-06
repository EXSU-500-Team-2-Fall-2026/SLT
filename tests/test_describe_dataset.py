import pandas as pd

from src.data import describe_dataset as dd


def _images():
    # L1 has two images of one melanoma; L2 is a nevus recorded at two sites;
    # L3 has no age, sex or site.
    return pd.DataFrame({
        "image_id": ["A", "B", "C", "D", "E"],
        "lesion_id": ["L1", "L1", "L2", "L2", "L3"],
        "dx": ["mel", "mel", "nv", "nv", "nv"],
        "label": [1, 1, 0, 0, 0],
        "age": [60.0, 60.0, 40.0, 40.0, None],
        "sex": ["male", "male", "female", "female", "unknown"],
        "localization": ["back", "back", "face", "neck", "unknown"],
    })


def test_counts_images_and_lesions_separately():
    summary = dd.split_summary("x", _images())
    assert (summary["images"], summary["lesions"]) == (5, 3)
    assert summary["malignant_images_pct"] == 40.0
    assert summary["malignant_lesions_pct"] == 33.33


def test_demographics_are_per_lesion():
    summary = dd.split_summary("x", _images())
    assert summary["age_median"] == 50.0
    assert summary["lesions_missing_age"] == 1
    assert (summary["male_lesions"], summary["female_lesions"], summary["unknown_sex_lesions"]) == (1, 1, 1)
    assert summary["unknown_site_lesions"] == 1
    assert summary["lesions_with_several_sites"] == 1


def test_lesion_with_two_sites_is_counted_at_each():
    sites = dd.body_site_counts("x", _images()).set_index("site")["lesions"]
    assert sites["face"] == 1 and sites["neck"] == 1 and sites["back"] == 1 and sites["unknown"] == 1


def test_conflicting_age_is_flagged():
    images = _images()
    images.loc[1, "age"] = 65.0
    assert dd.split_summary("x", images)["lesions_with_conflicting_age"] == 1


def test_withdrawn_test_image_is_excluded(tmp_path):
    columns = ["image_id", "lesion_id", "dx", "label", "split"]
    pd.DataFrame([["A", "L1", "nv", 0, "train"]], columns=columns).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame([["B", "L2", "nv", 0, "val"]], columns=columns).to_csv(tmp_path / "val.csv", index=False)
    pd.DataFrame(
        [["ISIC_0035068", "T1", "nv", 0, "test"], ["C", "T2", "mel", 1, "test"]], columns=columns
    ).to_csv(tmp_path / "test.csv", index=False)
    groups = dd.load_groups(tmp_path)
    assert list(groups["test"]["image_id"]) == ["C"]
    assert len(groups["ham10000"]) == 2
    counts = dd.class_counts(groups)
    assert counts.query("group == 'test' and dx == 'mel'")["images"].item() == 1
