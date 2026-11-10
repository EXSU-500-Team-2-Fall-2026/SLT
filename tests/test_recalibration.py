import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.eval import recalibration as rc


def _overconfident_sample(n=4000, seed=0):
    """True risk p_true; the 'model' reports probabilities shifted upwards,
    like a classifier trained with balanced class weights."""
    rng = np.random.default_rng(seed)
    true_logit = rng.normal(-1.5, 1.2, n)
    y = rng.binomial(1, 1 / (1 + np.exp(-true_logit)))
    reported = 1 / (1 + np.exp(-(true_logit + 1.5)))
    return y, reported


def test_recovers_known_shift():
    y, p = _overconfident_sample()
    cal = rc.fit_logistic_recalibration(y, p)
    assert cal["converged"]
    assert cal["intercept"] == pytest.approx(-1.5, abs=0.15)
    assert cal["slope"] == pytest.approx(1.0, abs=0.1)


def test_improves_brier_on_fresh_data_and_keeps_ranking():
    y_fit, p_fit = _overconfident_sample(seed=0)
    y_new, p_new = _overconfident_sample(seed=1)
    cal = rc.fit_logistic_recalibration(y_fit, p_fit)
    p_cal = rc.apply_recalibration(p_new, cal)

    assert brier_score_loss(y_new, p_cal) < brier_score_loss(y_new, p_new)
    assert roc_auc_score(y_new, p_cal) == pytest.approx(roc_auc_score(y_new, p_new))
    assert abs(p_cal.mean() - y_new.mean()) < abs(p_new.mean() - y_new.mean())


def test_handles_probabilities_of_exactly_zero_and_one():
    y, p = _overconfident_sample(500)
    p[:5], p[5:10] = 0.0, 1.0
    cal = rc.fit_logistic_recalibration(y, p)
    out = rc.apply_recalibration([0.0, 0.5, 1.0], cal)
    assert np.isfinite(out).all()
    assert ((out > 0) & (out < 1)).all()


def test_refuses_too_few_cases_in_a_class():
    with pytest.raises(ValueError, match="too few"):
        rc.fit_logistic_recalibration([0] * 50 + [1] * 3, np.linspace(0.1, 0.9, 53))


def test_refuses_missing_probabilities():
    y, p = _overconfident_sample(100)
    p[0] = np.nan
    with pytest.raises(ValueError):
        rc.fit_logistic_recalibration(y, p)


def test_save_and_load_round_trip(tmp_path):
    y, p = _overconfident_sample(500)
    cal = rc.fit_logistic_recalibration(y, p)
    rc.save_calibrator(cal, tmp_path / "models" / "cnn_calibrator.json")
    loaded = rc.load_calibrator(tmp_path / "models" / "cnn_calibrator.json")
    assert np.allclose(rc.apply_recalibration(p, loaded), rc.apply_recalibration(p, cal))
