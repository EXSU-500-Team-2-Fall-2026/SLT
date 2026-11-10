"""Post-hoc logistic recalibration of predicted probabilities.

Implements step 3 of the "Future recalibration protocol" in
reports/experimental-design.md:

    p_cal = sigmoid(intercept + slope * logit(p_original))

One mapping is fitted per frozen model, on the calibration subset only
(data/splits_v2/calibration.csv), with the observed outcomes and no class
weights or resampling. The intercept shifts the overall probability level;
the slope stretches or shrinks the spread of the log-odds. The mapping is
monotonic when slope > 0, so it cannot change the ranking of cases and
leaves AUROC unchanged.

Numerical settings, recorded with every fitted calibrator:
- probabilities are clipped to [eps, 1 - eps] with eps = 1e-6 before the
  logit (the same rule as src/eval/metrics.py);
- the logistic regression is unpenalised (C = inf), lbfgs solver.

This module only fits, saves, loads and applies the mapping. Fitting it on
the calibration subset says nothing about whether calibration improved;
that is assessed once, on the held-out cohort.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-6
MIN_CASES_PER_CLASS = 10


def _logit(p, eps: float = EPS) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_logistic_recalibration(y_true, y_prob, eps: float = EPS) -> dict:
    """Fit intercept and slope on calibration cases. Returns a plain dict
    that can be saved as JSON and passed to apply_recalibration."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same length")
    if np.isnan(y_prob).any():
        raise ValueError("y_prob contains missing values")

    n_pos, n_neg = int(y_true.sum()), int((1 - y_true).sum())
    if min(n_pos, n_neg) < MIN_CASES_PER_CLASS:
        raise ValueError(
            f"too few calibration cases in one class (positive={n_pos}, negative={n_neg}; "
            f"need at least {MIN_CASES_PER_CLASS} of each)"
        )

    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    model.fit(_logit(y_prob, eps).reshape(-1, 1), y_true)
    converged = bool(model.n_iter_[0] < model.max_iter)

    return {
        "method": "logistic recalibration: sigmoid(intercept + slope * logit(p))",
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0, 0]),
        "eps": eps,
        "solver": "lbfgs",
        "regularization": "none (C=inf)",
        "class_weight": "none",
        "converged": converged,
        "n_calibration": int(len(y_true)),
        "n_positive": n_pos,
    }


def apply_recalibration(y_prob, calibrator: dict) -> np.ndarray:
    z = calibrator["intercept"] + calibrator["slope"] * _logit(y_prob, calibrator.get("eps", EPS))
    return 1.0 / (1.0 + np.exp(-z))


def save_calibrator(calibrator: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(calibrator, f, indent=2)


def load_calibrator(path: Path) -> dict:
    with open(path) as f:
        calibrator = json.load(f)
    missing = {"intercept", "slope"} - set(calibrator)
    if missing:
        raise ValueError(f"{path} is missing {missing}")
    return calibrator
