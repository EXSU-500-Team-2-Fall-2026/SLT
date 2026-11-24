"""Score the held-out test set with the frozen models. This is run ONCE.

The held-out set (official ISIC 2018 Task 3 test images) has played no part
in fitting, model selection, calibrator fitting or threshold selection.
This script is the only code that scores it. It writes one file of
predictions and a record of exactly what produced them; all analysis is
then done from that file by src/eval/final_report.py, which never scores
an image and can be re-run freely.

Before scoring, the script checks that everything is frozen:
- each model file exists;
- the calibrator and threshold in models/ (the ones that would be applied)
  are identical to the committed records in <reports-dir>/<model>/.

It refuses to run if held-out predictions already exist.

Cohort and inputs:
- Every image in the test manifest whose file is present is scored by all
  three models, so the models are compared on the same cases. Images
  without a file are excluded and listed (one image, ISIC_0035068, was
  withdrawn from the distributed collection).
- The diagnosis in the ground-truth table must agree with the manifest,
  and map to the same binary label, for every image.
- Missing age, sex or body site are handled by the metadata model's own
  fitted imputation (training median; "unknown" category). Nothing is
  fitted on test data.
- The CNN and the hand-crafted features use the same preprocessing code as
  in training.

Rehearsal: `--rehearsal` runs the identical scoring path on the tuning
subset and compares the result with the predictions saved at training
time. It must reproduce them. Run it, and the analysis rehearsal in
final_report, before the real run, so that no code is debugged on
held-out data.

Usage:
    python -m src.eval.score_heldout --rehearsal      # tuning subset; checks the scoring path
    python -m src.eval.score_heldout                  # the one-time held-out run
    python -m src.eval.final_report --predictions reports/v2/final/predictions_test.csv --out reports/v2/final
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.labels import dx_to_binary
from src.eval.final_report import MODEL_NAMES, add_calibrated_and_flag
from src.eval.freeze_thresholds import load_threshold
from src.eval.recalibration import load_calibrator
from src.eval.reporting import HELD_OUT_SPLIT
from src.features.handcrafted import FEATURE_NAMES, extract_features_from_path
from src.models.metadata_baseline import FEATURE_COLUMNS
from src.predict import MODEL_FILES, load_models

REHEARSAL_SPLIT = "tune"
REHEARSAL_TOLERANCE = 1e-5
SOURCES = {
    # split -> (image folder, metadata table) under the raw-data folder
    HELD_OUT_SPLIT: ("ISIC2018_Task3_Test_Images", "ISIC2018_Task3_Test_GroundTruth.tab"),
    REHEARSAL_SPLIT: ("HAM10000_images", "HAM10000_metadata.tab"),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_frozen(models: list[str], models_dir: Path, reports_dir: Path) -> tuple[dict, list[str]]:
    """Returns (record of what is frozen, list of problems). Scoring must
    not start while the list is non-empty."""
    record, problems = {}, []
    for model in models:
        model_path = Path(models_dir) / MODEL_FILES[model]
        if not model_path.exists():
            problems.append(f"{model}: model file {model_path} not found")
            continue
        entry = {"model_file": model_path.as_posix(), "model_sha256": file_sha256(model_path)}
        for kind, loader in (("calibrator", load_calibrator), ("threshold", load_threshold)):
            used_path = Path(models_dir) / f"{model}_{kind}.json"
            recorded_path = Path(reports_dir) / model / f"{kind}.json"
            if not used_path.exists() or not recorded_path.exists():
                problems.append(f"{model}: {kind} missing ({used_path} and {recorded_path} must both exist)")
                continue
            used, recorded = loader(used_path), loader(recorded_path)
            if used != recorded:
                problems.append(f"{model}: {used_path} differs from the committed record {recorded_path}")
                continue
            entry[kind] = recorded
        record[model] = entry
    return record, problems


def load_cohort(split: str, splits_dir: Path, raw_dir: Path) -> tuple[pd.DataFrame, Path, list[str]]:
    """Returns (rows to score, with age/sex/localization and image source;
    image folder; image_ids excluded because their file is absent)."""
    image_folder, table_name = SOURCES[split]
    image_dir = Path(raw_dir) / image_folder
    manifest = pd.read_csv(Path(splits_dir) / f"{split}.csv")
    table = pd.read_csv(Path(raw_dir) / table_name, sep=None, engine="python")

    columns = ["image_id", "dx", *FEATURE_COLUMNS] + (["dataset"] if "dataset" in table.columns else [])
    merged = manifest.merge(table[columns].rename(columns={"dx": "dx_in_table"}), on="image_id", how="left",
                            validate="one_to_one")

    not_in_table = merged.loc[merged["dx_in_table"].isna(), "image_id"].tolist()
    if not_in_table:
        raise ValueError(f"{len(not_in_table)} manifest image(s) are not in {table_name}, e.g. {not_in_table[:3]}")
    disagree = merged[merged["dx_in_table"].str.strip().str.lower() != merged["dx"].str.strip().str.lower()]
    if len(disagree):
        raise ValueError(f"diagnosis differs between manifest and {table_name} for {len(disagree)} image(s)")
    wrong_label = merged[merged["dx"].map(dx_to_binary) != merged["label"]]
    if len(wrong_label):
        raise ValueError(f"binary label does not follow from the diagnosis for {len(wrong_label)} image(s)")
    if merged["image_id"].duplicated().any():
        raise ValueError("duplicate image_id in the manifest")

    present = np.array([(image_dir / f"{image_id}.jpg").exists() for image_id in merged["image_id"]])
    excluded = merged.loc[~present, "image_id"].tolist()
    cohort = merged.loc[present].drop(columns="dx_in_table").reset_index(drop=True)
    return cohort, image_dir, excluded


def score_metadata(model, cohort: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(cohort[FEATURE_COLUMNS])[:, 1]


def score_classical(model, cohort: pd.DataFrame, image_dir: Path, feature_pool: pd.DataFrame | None
                    ) -> tuple[np.ndarray, pd.DataFrame]:
    """Uses cached feature rows where they exist (development images) and
    extracts the rest with the training feature code."""
    cached = {} if feature_pool is None else feature_pool.set_index("image_id")[FEATURE_NAMES].to_dict("index")
    rows, extracted = [], 0
    for position, image_id in enumerate(cohort["image_id"], start=1):
        if image_id in cached:
            rows.append(cached[image_id])
        else:
            rows.append(extract_features_from_path(image_dir / f"{image_id}.jpg"))
            extracted += 1
            if extracted % 200 == 0:
                print(f"  hand-crafted features: {position} of {len(cohort)} images", flush=True)
    features = pd.DataFrame(rows, columns=FEATURE_NAMES)
    probabilities = model.predict_proba(features)[:, 1]
    features.insert(0, "image_id", cohort["image_id"].to_numpy())
    return probabilities, features


def score_cnn(model, cohort: pd.DataFrame, image_dir: Path, image_size: int, device_name: str,
              batch_size: int = 64) -> tuple[np.ndarray, str]:
    from torch.utils.data import DataLoader

    from src.models.cnn import HAM10000ImageDataset, get_transforms, pick_device, predict_probs

    device = pick_device(device_name)
    with tempfile.TemporaryDirectory() as folder:
        manifest_path = Path(folder) / "cohort.csv"
        cohort[["image_id", "label"]].to_csv(manifest_path, index=False)
        dataset = HAM10000ImageDataset(manifest_path, image_dir, get_transforms(image_size, train=False))
    if list(dataset.df["image_id"]) != list(cohort["image_id"]):
        raise RuntimeError("the CNN dataset does not match the cohort")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    _, probabilities = predict_probs(model.to(device), loader, device)
    model.to("cpu")
    return probabilities.astype(float), str(device)


def git_state() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
        changed = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                 text=True, timeout=10).stdout.strip()
        return {"commit": commit or None, "uncommitted_changes_to_tracked_files": bool(changed)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "uncommitted_changes_to_tracked_files": None}


def compare_with_saved(table: pd.DataFrame, models: list[str], reports_dir: Path, split: str) -> dict:
    """Largest absolute difference, per model, between these raw
    probabilities and the ones saved when the model was trained."""
    differences = {}
    for model in models:
        saved = pd.read_csv(Path(reports_dir) / model / f"predictions_{split}.csv")[["image_id", "prob"]]
        both = table[["image_id", f"{model}_raw"]].merge(saved, on="image_id", how="inner", validate="one_to_one")
        if len(both) != len(table):
            raise ValueError(f"{model}: saved {split} predictions do not cover the same images")
        differences[model] = float(np.abs(both[f"{model}_raw"] - both["prob"]).max())
    return differences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rehearsal", action="store_true", help="score the tuning subset and check it against saved predictions")
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits_v2"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/v2"))
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--out", type=Path, default=None, help="default: <reports-dir>/final or <reports-dir>/final_rehearsal")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    args = parser.parse_args()

    split = REHEARSAL_SPLIT if args.rehearsal else HELD_OUT_SPLIT
    out_dir = args.out or args.reports_dir / ("final_rehearsal" if args.rehearsal else "final")
    predictions_path = out_dir / f"predictions_{split}.csv"
    if not args.rehearsal and predictions_path.exists():
        sys.exit(f"{predictions_path} already exists. The held-out set is scored once; analyse that file with "
                 "src/eval/final_report.py.")

    frozen, problems = check_frozen(args.models, args.models_dir, args.reports_dir)
    if problems:
        sys.exit("Not everything is frozen:\n  - " + "\n  - ".join(problems))
    loaded = load_models(args.models_dir, args.reports_dir)
    unavailable = [m for m in args.models if m not in loaded["models"]]
    if unavailable:
        sys.exit("Could not load: " + ", ".join(f"{m} ({loaded['skipped'].get(m)})" for m in unavailable))

    try:
        cohort, image_dir, excluded = load_cohort(split, args.splits_dir, args.raw_dir)
    except (ValueError, FileNotFoundError) as problem:
        sys.exit(f"Cohort check failed: {problem}")
    if len(cohort) == 0:
        sys.exit(f"No images found in {image_dir}. See 'Getting the data' in README.md.")
    print(f"{split}: {len(cohort)} images to score, {len(excluded)} excluded (no image file): {excluded[:5]}")

    table = cohort[["image_id", "lesion_id", "group_id", "dx", "label", "split"]
                   + (["dataset"] if "dataset" in cohort.columns else [])].copy()
    device_used, features = None, None
    for model in args.models:
        print(f"scoring with {model} ...", flush=True)
        if model == "metadata_baseline":
            raw = score_metadata(loaded["models"][model], cohort)
        elif model == "classical_ml":
            from src.models.classical_ml import load_feature_pool

            pool = load_feature_pool(args.features_dir) if any(Path(args.features_dir).glob("*.csv")) else None
            raw, features = score_classical(loaded["models"][model], cohort, image_dir, pool)
        else:
            raw, device_used = score_cnn(loaded["models"][model], cohort, image_dir, loaded["cnn_image_size"], args.device)
        if len(raw) != len(table) or not np.isfinite(raw).all():
            sys.exit(f"{model}: scoring did not return one finite probability per image")
        table[f"{model}_raw"] = raw
        add_calibrated_and_flag(table, model, frozen[model]["calibrator"], frozen[model]["threshold"])

    record = {
        "purpose": "rehearsal of the scoring path on the tuning subset" if args.rehearsal else "held-out evaluation",
        "split": split,
        "scored_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest_images": int(len(cohort) + len(excluded)),
        "scored_images": int(len(cohort)),
        "scored_lesions": int(cohort["lesion_id"].nunique()),
        "excluded_images": [{"image_id": i, "reason": "image file not in the distributed collection"} for i in excluded],
        "missing_metadata": {c: int(cohort[c].isna().sum()) for c in FEATURE_COLUMNS},
        "missing_metadata_handling": "the metadata model's fitted imputation (training median age; 'unknown' category)",
        "cnn_device": device_used,
        "frozen": frozen,
        "code": git_state(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }

    if args.rehearsal:
        differences = compare_with_saved(table, args.models, args.reports_dir, split)
        record["largest_difference_from_saved_predictions"] = differences
        for model, difference in differences.items():
            print(f"{model:<20} largest difference from the saved {split} predictions: {difference:.2e}")
        if max(differences.values()) > REHEARSAL_TOLERANCE:
            sys.exit("Rehearsal FAILED: the scoring path does not reproduce the predictions saved at training time. "
                     "Do not run the held-out evaluation until this is understood.")
        print("Rehearsal passed: the scoring path reproduces the saved predictions.")

    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(predictions_path, index=False)
    with open(out_dir / "run_record.json", "w") as f:
        json.dump(record, f, indent=2)
    if features is not None and not args.rehearsal:
        features.to_csv(out_dir / f"features_{split}.csv", index=False)

    print(f"wrote {predictions_path} and {out_dir / 'run_record.json'}")
    if args.rehearsal:
        print(f"Next: python -m src.eval.final_report --from-saved {split} --out {out_dir}")
    else:
        print(f"Next: python -m src.eval.final_report --predictions {predictions_path} --out {out_dir}")


if __name__ == "__main__":
    main()
