"""Discrimination and calibration metrics shared by all three models
(metadata baseline, classical ML, CNN) so their results are directly
comparable.

- Discrimination: AUROC, AUPRC, sensitivity/specificity at a stated
  threshold.
- Calibration: Brier score, expected calibration error (ECE), and
  calibration slope/intercept from a Cox calibration regression
  (refitting label ~ logit(predicted probability)); slope near 1 and
  intercept near 0 indicate good calibration.
- Variability: bootstrap_metrics_ci resamples the validation set with
  replacement to give a 95% CI on every metric above, since a point
  estimate with no uncertainty isn't a result (course spec section 5.2).
"""
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score


@dataclass
class DiscriminationMetrics:
    auroc: float
    auprc: float
    sensitivity: float
    specificity: float
    threshold: float


@dataclass
class CalibrationMetrics:
    brier_score: float
    ece: float
    calibration_slope: float
    calibration_intercept: float
    bin_edges: np.ndarray
    bin_confidence: np.ndarray
    bin_accuracy: np.ndarray
    bin_counts: np.ndarray


def discrimination_metrics(y_true, y_prob, threshold: float = 0.5) -> DiscriminationMetrics:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return DiscriminationMetrics(auroc, auprc, sensitivity, specificity, threshold)


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bin_edges[1:-1], right=True), 0, n_bins - 1)

    bin_confidence = np.full(n_bins, np.nan)
    bin_accuracy = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        bin_counts[b] = count
        if count == 0:
            continue
        bin_confidence[b] = y_prob[mask].mean()
        bin_accuracy[b] = y_true[mask].mean()
        ece += (count / n) * abs(bin_accuracy[b] - bin_confidence[b])

    return ece, bin_edges, bin_confidence, bin_accuracy, bin_counts


def _calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-6):
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    logit_p = np.log(y_prob_clipped / (1 - y_prob_clipped)).reshape(-1, 1)
    model = LogisticRegression()
    model.fit(logit_p, y_true)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def calibration_metrics(y_true, y_prob, n_bins: int = 10) -> CalibrationMetrics:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    brier = brier_score_loss(y_true, y_prob)
    ece, bin_edges, bin_confidence, bin_accuracy, bin_counts = _expected_calibration_error(
        y_true, y_prob, n_bins
    )
    slope, intercept = _calibration_slope_intercept(y_true, y_prob)
    return CalibrationMetrics(
        brier_score=brier,
        ece=ece,
        calibration_slope=slope,
        calibration_intercept=intercept,
        bin_edges=bin_edges,
        bin_confidence=bin_confidence,
        bin_accuracy=bin_accuracy,
        bin_counts=bin_counts,
    )


def bootstrap_metrics_ci(
    y_true, y_prob, threshold: float = 0.5, n_boot: int = 1000, ci: float = 0.95, seed: int = 42
) -> dict:
    """Resample (y_true, y_prob) with replacement n_boot times and return a
    (lower, upper) percentile interval for each metric. Resamples with only
    one class present are skipped (AUROC/AUPRC are undefined there)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    samples = {"auroc": [], "auprc": [], "sensitivity": [], "specificity": [], "brier_score": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        disc = discrimination_metrics(yt, yp, threshold)
        samples["auroc"].append(disc.auroc)
        samples["auprc"].append(disc.auprc)
        samples["sensitivity"].append(disc.sensitivity)
        samples["specificity"].append(disc.specificity)
        samples["brier_score"].append(brier_score_loss(yt, yp))

    alpha = (1.0 - ci) / 2.0
    result = {
        f"{name}_ci": [
            float(np.percentile(values, 100 * alpha)),
            float(np.percentile(values, 100 * (1 - alpha))),
        ]
        for name, values in samples.items()
    }
    result["n_boot"] = len(samples["auroc"])
    result["ci_level"] = ci
    return result


def reliability_diagram(calibration: CalibrationMetrics, title: str, out_path) -> None:
    import matplotlib.pyplot as plt

    bin_centers = (calibration.bin_edges[:-1] + calibration.bin_edges[1:]) / 2
    observed = np.where(np.isnan(calibration.bin_accuracy), bin_centers, calibration.bin_accuracy)
    predicted = np.where(np.isnan(calibration.bin_confidence), bin_centers, calibration.bin_confidence)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(predicted, observed, marker="o", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction malignant")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
