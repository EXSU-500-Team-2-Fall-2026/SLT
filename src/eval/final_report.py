"""Analyse a file of frozen-model predictions: the prespecified comparison
of original and recalibrated probabilities, the model comparison, and
their uncertainty.

This script only reads predictions that already exist. It never scores an
image, so it can be run and re-run freely. Scoring the held-out set is a
separate, one-time step (src/eval/score_heldout.py).

Follows "Compare original and recalibrated probabilities on the same
held-out cases" in reports/experimental-design.md:

- Primary before/after outcome: Brier score, recalibrated minus original;
  negative favours recalibration.
- Supporting: reliability diagrams and calibration slope/intercept before
  and after; ECE with 10 equal-width bins.
- Discrimination: AUROC and average precision (AUPRC). Recalibration with a
  positive slope keeps the ranking, so these do not change.
- Operating point: sensitivity and specificity at each model's frozen
  threshold (src/eval/freeze_thresholds.py). Decisions are made on the raw
  probability, so they are the same before and after recalibration.
- Uncertainty: paired group bootstrap. Whole groups (group_id: a lesion,
  or lesions linked by known near-duplicate images) are resampled with
  replacement, keeping all their images; the same resampled groups are
  used for every model and for before and after. 95% percentile intervals.
  The intervals condition on the frozen models, calibrators and
  thresholds; they do not capture what would change if the models were
  retrained.
- Sensitivity by diagnosis, and performance by image source, as
  descriptive breakdowns.

The verdict for each model follows the protocol's three outcomes: if the
whole interval of the Brier difference is below zero, "improvement"; if it
is above zero, "deterioration"; otherwise "no clear evidence of change".

Usage:
    # development rehearsal on the saved tuning-subset predictions
    python -m src.eval.final_report --from-saved tune --out reports/v2/final_rehearsal

    # the held-out analysis, after score_heldout has written the predictions
    python -m src.eval.final_report --predictions reports/v2/final/predictions_test.csv --out reports/v2/final
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.eval.metrics import calibration_metrics, reliability_points
from src.eval.recalibration import apply_recalibration, load_calibrator
from src.eval.reporting import HELD_OUT_SPLIT

MODEL_NAMES = ("metadata_baseline", "classical_ml", "cnn")
MODEL_LABELS = {
    "metadata_baseline": "Metadata baseline",
    "classical_ml": "Classical ML",
    "cnn": "CNN",
}
N_BOOT = 2000
SEED = 42
CI_LEVEL = 0.95
MIN_SOURCE_POSITIVES = 10  # AUROC by image source is reported only above this many positive and negative images


# ---------------------------------------------------------------- inputs

def models_in(predictions: pd.DataFrame) -> list[str]:
    return [m for m in MODEL_NAMES if f"{m}_raw" in predictions.columns]


def assemble_from_saved(reports_dir: Path, split: str, splits_dir: Path, metadata_path: Path | None = None) -> pd.DataFrame:
    """Build the predictions table for a development split from the files
    the training scripts saved, plus the fitted calibrators and frozen
    thresholds. Used to rehearse the analysis without touching the
    held-out set."""
    if split == HELD_OUT_SPLIT:
        raise ValueError("held-out predictions are produced only by src/eval/score_heldout.py")
    from src.eval.freeze_thresholds import load_threshold

    manifest = pd.read_csv(Path(splits_dir) / f"{split}.csv")
    table = manifest[["image_id", "lesion_id", "group_id", "dx", "label", "split"]].copy()
    if metadata_path is not None and Path(metadata_path).exists():
        meta = pd.read_csv(metadata_path, sep=None, engine="python")
        if "dataset" in meta.columns:
            table = table.merge(meta[["image_id", "dataset"]], on="image_id", how="left", validate="one_to_one")

    for model in MODEL_NAMES:
        folder = Path(reports_dir) / model
        saved = folder / f"predictions_{split}.csv"
        if not saved.exists():
            continue
        raw = pd.read_csv(saved)[["image_id", "prob"]].rename(columns={"prob": f"{model}_raw"})
        table = table.merge(raw, on="image_id", how="left", validate="one_to_one")
        if table[f"{model}_raw"].isna().any():
            raise ValueError(f"{saved} does not cover every image in the {split} manifest")
        add_calibrated_and_flag(table, model, load_calibrator(folder / "calibrator.json"),
                                load_threshold(folder / "threshold.json"))
    return table


def add_calibrated_and_flag(table: pd.DataFrame, model: str, calibrator: dict, threshold: dict) -> None:
    """Add <model>_recalibrated and <model>_flagged in place. The flag
    compares the raw probability with the frozen raw threshold."""
    raw = table[f"{model}_raw"].to_numpy()
    table[f"{model}_recalibrated"] = apply_recalibration(raw, calibrator)
    table[f"{model}_flagged"] = (raw >= threshold["threshold_raw"]).astype(int)


# ---------------------------------------------------------------- metrics

def model_metrics(y: np.ndarray, raw: np.ndarray, recalibrated: np.ndarray, flagged: np.ndarray) -> dict:
    """All metrics for one model on one set of cases (a flat dict)."""
    positive = y == 1
    before, after = calibration_metrics(y, raw), calibration_metrics(y, recalibrated)
    flagged = flagged.astype(bool)
    return {
        "auroc": float(roc_auc_score(y, raw)),
        "auroc_recalibrated": float(roc_auc_score(y, recalibrated)),
        "auprc": float(average_precision_score(y, raw)),
        "auprc_recalibrated": float(average_precision_score(y, recalibrated)),
        "sensitivity": float((flagged & positive).sum() / positive.sum()),
        "specificity": float((~flagged & ~positive).sum() / (~positive).sum()),
        "flagged_fraction": float(flagged.mean()),
        "brier_original": float(before.brier_score),
        "brier_recalibrated": float(after.brier_score),
        "brier_difference": float(after.brier_score - before.brier_score),
        "ece_original": float(before.ece),
        "ece_recalibrated": float(after.ece),
        "ece_difference": float(after.ece - before.ece),
        "calibration_slope_original": float(before.calibration_slope),
        "calibration_slope_recalibrated": float(after.calibration_slope),
        "calibration_intercept_original": float(before.calibration_intercept),
        "calibration_intercept_recalibrated": float(after.calibration_intercept),
        "mean_predicted_original": float(raw.mean()),
        "mean_predicted_recalibrated": float(recalibrated.mean()),
    }


PAIRED_METRICS = ("auroc", "auprc", "sensitivity", "specificity", "brier_recalibrated")


def all_metrics(predictions: pd.DataFrame, models: list[str]) -> dict:
    """{model: metrics} plus {"<a>_minus_<b>": differences} for each pair of models."""
    y = predictions["label"].to_numpy().astype(int)
    result = {
        m: model_metrics(y, predictions[f"{m}_raw"].to_numpy(), predictions[f"{m}_recalibrated"].to_numpy(),
                         predictions[f"{m}_flagged"].to_numpy())
        for m in models
    }
    for first, second in combinations(models, 2):  # MODEL_NAMES order: simpler model first
        result[f"{second}_minus_{first}"] = {k: result[second][k] - result[first][k] for k in PAIRED_METRICS}
    return result


def group_bootstrap(predictions: pd.DataFrame, models: list[str], n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Percentile intervals from resampling whole groups with replacement.
    Returns {key: {metric: [low, high]}} with the same keys as all_metrics,
    plus "_n_boot" (resamples used; those with one class only are skipped)."""
    groups = predictions["group_id"].to_numpy()
    _, inverse = np.unique(groups, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    boundaries = np.flatnonzero(np.diff(inverse[order])) + 1
    members = np.split(order, boundaries)  # row positions of each group
    n_groups = len(members)

    rng = np.random.default_rng(seed)
    draws: dict = {}
    used = 0
    for _ in range(n_boot):
        chosen = rng.integers(0, n_groups, n_groups)
        rows = np.concatenate([members[g] for g in chosen])
        sample = predictions.iloc[rows]
        if sample["label"].nunique() < 2:
            continue
        used += 1
        for key, values in all_metrics(sample, models).items():
            store = draws.setdefault(key, {})
            for metric, value in values.items():
                store.setdefault(metric, []).append(value)

    alpha = (1 - CI_LEVEL) / 2
    intervals = {
        key: {metric: [float(np.percentile(v, 100 * alpha)), float(np.percentile(v, 100 * (1 - alpha)))]
              for metric, v in store.items()}
        for key, store in draws.items()
    }
    intervals["_n_boot"] = used
    return intervals


def verdict(interval: list[float]) -> str:
    low, high = interval
    if high < 0:
        return "improvement"
    if low > 0:
        return "deterioration"
    return "no clear evidence of change"


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def flag_rate_by_diagnosis(predictions: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Share of images flagged, per diagnosis and model. For malignant or
    premalignant diagnoses this is sensitivity; for benign ones it is the
    false-positive rate. Wilson intervals treat images as independent."""
    rows = []
    for (dx, label), part in predictions.groupby(["dx", "label"]):
        for m in models:
            flagged = int(part[f"{m}_flagged"].sum())
            low, high = wilson_interval(flagged, len(part))
            rows.append({"dx": dx, "group": "malignant/premalignant" if label == 1 else "benign", "model": m,
                         "images": len(part), "flagged": flagged, "flag_rate": flagged / len(part),
                         "wilson_low": low, "wilson_high": high})
    return pd.DataFrame(rows).sort_values(["group", "dx", "model"], ascending=[False, True, True]).reset_index(drop=True)


def performance_by_source(predictions: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Descriptive breakdown by image source ('dataset' column), without
    intervals. AUROC is given only where a source has enough images of
    both classes."""
    if "dataset" not in predictions.columns:
        return pd.DataFrame()
    rows = []
    for source, part in predictions.fillna({"dataset": "unknown"}).groupby("dataset"):
        y = part["label"].to_numpy().astype(int)
        n_pos, n_neg = int(y.sum()), int((1 - y).sum())
        for m in models:
            flagged = part[f"{m}_flagged"].to_numpy().astype(bool)
            enough = min(n_pos, n_neg) >= MIN_SOURCE_POSITIVES
            rows.append({
                "source": source, "model": m, "images": len(part), "positive": n_pos,
                "auroc": float(roc_auc_score(y, part[f"{m}_raw"])) if enough else np.nan,
                "sensitivity": float(flagged[y == 1].mean()) if n_pos else np.nan,
                "specificity": float((~flagged[y == 0]).mean()) if n_neg else np.nan,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- outputs

def reliability_figure(predictions: pd.DataFrame, models: list[str], title: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = predictions["label"].to_numpy().astype(int)
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.4), squeeze=False)
    for ax, m in zip(axes[0], models):
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        for column, name, marker in ((f"{m}_raw", "Original", "o"), (f"{m}_recalibrated", "Recalibrated", "s")):
            predicted, observed, _ = reliability_points(calibration_metrics(y, predictions[column].to_numpy()))
            ax.plot(predicted, observed, marker=marker, label=name)
        ax.set_title(MODEL_LABELS[m])
        ax.set_xlabel("Mean predicted probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0][0].set_ylabel("Observed fraction malignant/premalignant")
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _fmt(value: float, interval: list[float] | None = None, digits: int = 3) -> str:
    text = f"{value:.{digits}f}"
    return text if interval is None else f"{text} [{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]"


def summary_markdown(cohort: dict, metrics: dict, intervals: dict, models: list[str], by_dx: pd.DataFrame) -> str:
    lines = [
        f"# Frozen-model evaluation: {cohort['split']} subset",
        "",
        f"{cohort['images']} images, {cohort['lesions']} lesions, {cohort['groups']} resampling groups, "
        f"{cohort['positive_images']} malignant/premalignant ({100 * cohort['positive_images'] / cohort['images']:.2f}%).",
        f"95% percentile intervals from {intervals['_n_boot']} paired group-bootstrap resamples (seed {SEED}). "
        "They condition on the frozen models, calibrators and thresholds.",
        "",
        "## Discrimination and operating point",
        "",
        "| Model | AUROC | AUPRC | Sensitivity | Specificity | Flagged |",
        "|---|---|---|---|---|---|",
    ]
    for m in models:
        v, ci = metrics[m], intervals[m]
        lines.append(f"| {MODEL_LABELS[m]} | {_fmt(v['auroc'], ci['auroc'])} | {_fmt(v['auprc'], ci['auprc'])} | "
                     f"{_fmt(v['sensitivity'], ci['sensitivity'])} | {_fmt(v['specificity'], ci['specificity'])} | "
                     f"{v['flagged_fraction']:.2f} |")
    lines += ["", "Sensitivity and specificity are at each model's frozen threshold.", "",
              "## Calibration before and after recalibration", "",
              "| Model | Brier original | Brier recalibrated | Difference (primary) | Verdict | ECE original | ECE recalibrated | "
              "Slope / intercept original | Slope / intercept recalibrated |",
              "|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        v, ci = metrics[m], intervals[m]
        lines.append(
            f"| {MODEL_LABELS[m]} | {_fmt(v['brier_original'], ci['brier_original'])} | "
            f"{_fmt(v['brier_recalibrated'], ci['brier_recalibrated'])} | "
            f"{_fmt(v['brier_difference'], ci['brier_difference'], 4)} | {verdict(ci['brier_difference'])} | "
            f"{_fmt(v['ece_original'], ci['ece_original'])} | {_fmt(v['ece_recalibrated'], ci['ece_recalibrated'])} | "
            f"{v['calibration_slope_original']:.2f} / {v['calibration_intercept_original']:.2f} | "
            f"{v['calibration_slope_recalibrated']:.2f} / {v['calibration_intercept_recalibrated']:.2f} |")
    lines += ["", "Difference is recalibrated minus original; negative favours recalibration.", "",
              "## Paired differences between models", "",
              "| Comparison | AUROC | AUPRC | Sensitivity | Specificity | Brier (recalibrated) |", "|---|---|---|---|---|---|"]
    for first, second in combinations(models, 2):
        key = f"{second}_minus_{first}"
        v, ci = metrics[key], intervals[key]
        lines.append(f"| {MODEL_LABELS[second]} minus {MODEL_LABELS[first]} | " + " | ".join(
            _fmt(v[k], ci[k]) for k in PAIRED_METRICS) + " |")
    lines += ["", "An interval that excludes zero indicates a difference on these cases; one that includes zero does not "
              "show that the models are equivalent.", "", "## Flag rate by diagnosis", "",
              "| Diagnosis | Group | Images | " + " | ".join(MODEL_LABELS[m] for m in models) + " |",
              "|---|---|---|" + "---|" * len(models)]
    for (dx, group), part in by_dx.groupby(["dx", "group"], sort=False):
        cells = []
        for m in models:
            row = part[part["model"] == m].iloc[0]
            cells.append(f"{row['flag_rate']:.3f} [{row['wilson_low']:.3f}, {row['wilson_high']:.3f}]")
        lines.append(f"| {dx} | {group} | {int(part['images'].iloc[0])} | " + " | ".join(cells) + " |")
    lines += ["", "For malignant/premalignant diagnoses the flag rate is sensitivity; for benign ones it is the "
              "false-positive rate. Wilson intervals, treating images as independent.", ""]
    return "\n".join(lines)


def write_report(predictions: pd.DataFrame, out_dir: Path, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = models_in(predictions)
    if not models:
        raise ValueError("the predictions file has no <model>_raw columns")
    split = str(predictions["split"].iloc[0]) if "split" in predictions.columns else "unknown"
    cohort = {
        "split": split,
        "images": int(len(predictions)),
        "lesions": int(predictions["lesion_id"].nunique()),
        "groups": int(predictions["group_id"].nunique()),
        "positive_images": int(predictions["label"].sum()),
    }

    metrics = all_metrics(predictions, models)
    intervals = group_bootstrap(predictions, models, n_boot=n_boot, seed=seed)
    by_dx = flag_rate_by_diagnosis(predictions, models)
    by_source = performance_by_source(predictions, models)

    report = {
        "cohort": cohort,
        "bootstrap": {"method": "paired group bootstrap, percentile intervals", "resampling_unit": "group_id",
                      "n_boot_requested": n_boot, "n_boot_used": intervals["_n_boot"], "seed": seed,
                      "ci_level": CI_LEVEL},
        "models": {m: {"estimate": metrics[m], "ci95": intervals[m],
                       "recalibration_verdict": verdict(intervals[m]["brier_difference"])} for m in models},
        "model_differences": {k: {"estimate": metrics[k], "ci95": intervals[k]}
                              for k in metrics if k not in models},
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    by_dx.to_csv(out_dir / "flag_rate_by_diagnosis.csv", index=False)
    if len(by_source):
        by_source.to_csv(out_dir / "performance_by_source.csv", index=False)
    reliability_figure(predictions, models, f"Reliability before and after recalibration ({split} subset)",
                       out_dir / "reliability_before_after.png")
    (out_dir / "summary.md").write_text(summary_markdown(cohort, metrics, intervals, models, by_dx))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, default=None, help="predictions file written by score_heldout")
    parser.add_argument("--from-saved", default=None, metavar="SPLIT",
                        help="rehearse on a development split, from the predictions saved by the training scripts")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/v2"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits_v2"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if (args.predictions is None) == (args.from_saved is None):
        parser.error("give exactly one of --predictions or --from-saved")
    if args.from_saved:
        try:
            predictions = assemble_from_saved(args.reports_dir, args.from_saved, args.splits_dir,
                                              args.raw_dir / "HAM10000_metadata.tab")
        except ValueError as problem:
            sys.exit(str(problem))
    else:
        predictions = pd.read_csv(args.predictions)

    report = write_report(predictions, args.out, n_boot=args.n_boot, seed=args.seed)
    print((args.out / "summary.md").read_text())
    print(f"wrote {args.out}/ (metrics.json, summary.md, flag_rate_by_diagnosis.csv, reliability_before_after.png)")
    for model, entry in report["models"].items():
        print(f"{model:<20} recalibration: {entry['recalibration_verdict']}")


if __name__ == "__main__":
    main()
