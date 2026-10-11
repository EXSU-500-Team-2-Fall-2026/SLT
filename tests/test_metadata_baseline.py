import pandas as pd

from src.models.metadata_baseline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_pipeline


def _synthetic_frame(n=200, seed=0):
    rng = pd.Series(range(n))
    sex = rng.map(lambda i: ["male", "female", "unknown"][i % 3])
    localization = rng.map(lambda i: ["back", "face", "scalp"][i % 3])
    age = rng.map(lambda i: 20.0 + (i % 60))
    age = age.where(rng % 10 != 0, other=float("nan"))  # some missing ages
    label = ((rng % 3 == 0) & (sex == "male")).astype(int)
    return pd.DataFrame(
        {"age": age, "sex": sex, "localization": localization, "label": label}
    )


def test_pipeline_fits_and_predicts_probabilities():
    df = _synthetic_frame()
    pipeline = build_pipeline()
    pipeline.fit(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df["label"])

    probs = pipeline.predict_proba(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    assert len(probs) == len(df)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_pipeline_handles_missing_age_and_unseen_category():
    df = _synthetic_frame()
    pipeline = build_pipeline()
    pipeline.fit(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df["label"])

    unseen = pd.DataFrame(
        {"age": [float("nan")], "sex": ["unknown"], "localization": ["never_seen_before"]}
    )
    probs = pipeline.predict_proba(unseen[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    assert probs.shape == (1,)
