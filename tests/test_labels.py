import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit

from src.data.labels import ALL_CLASSES, BENIGN_CLASSES, MALIGNANT_CLASSES, dx_to_binary


@pytest.mark.parametrize("dx", sorted(MALIGNANT_CLASSES))
def test_malignant_classes_map_to_one(dx):
    assert dx_to_binary(dx) == 1


@pytest.mark.parametrize("dx", sorted(BENIGN_CLASSES))
def test_benign_classes_map_to_zero(dx):
    assert dx_to_binary(dx) == 0


def test_unknown_class_raises():
    with pytest.raises(ValueError):
        dx_to_binary("not_a_real_code")


def test_all_classes_is_seven():
    assert len(ALL_CLASSES) == 7


def test_group_shuffle_split_has_no_lesion_leakage():
    # Synthetic lesions with a varying number of images each, to exercise
    # the same grouping logic used in make_splits.split_train_val.
    rows = []
    for lesion_idx in range(50):
        n_images = 1 + (lesion_idx % 3)
        for _ in range(n_images):
            rows.append({"lesion_id": f"lesion_{lesion_idx}", "label": lesion_idx % 2})
    df = pd.DataFrame(rows)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, val_idx = next(splitter.split(df, groups=df["lesion_id"]))

    train_lesions = set(df.iloc[train_idx]["lesion_id"])
    val_lesions = set(df.iloc[val_idx]["lesion_id"])
    assert not (train_lesions & val_lesions)
