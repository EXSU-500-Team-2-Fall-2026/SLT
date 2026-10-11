import numpy as np
import pandas as pd

from src.features.handcrafted import FEATURE_NAMES
from src.models.classical_ml import build_pipeline


def _synthetic_features(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    # make the label actually depend on a couple of features, so the model
    # has something real to learn rather than pure noise
    label = ((df["compactness"] + df["lesion_skin_contrast"]) > 0).astype(int)
    df["label"] = label
    return df


def test_pipeline_fits_and_predicts_probabilities():
    df = _synthetic_features()
    model = build_pipeline()
    model.fit(df[FEATURE_NAMES], df["label"])

    probs = model.predict_proba(df[FEATURE_NAMES])[:, 1]
    assert len(probs) == len(df)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_pipeline_learns_better_than_chance_on_separable_signal():
    train_df = _synthetic_features(seed=1)
    test_df = _synthetic_features(seed=2)

    model = build_pipeline()
    model.fit(train_df[FEATURE_NAMES], train_df["label"])
    probs = model.predict_proba(test_df[FEATURE_NAMES])[:, 1]

    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(test_df["label"], probs) > 0.7
