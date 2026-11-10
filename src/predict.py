"""Score ONE image with whichever of the three trained models are available.

This is the single-image path that the interface will call. It reuses the
exact preprocessing of the training scripts:

- metadata baseline: age / sex / localization -> models/metadata_baseline.joblib
- classical ML:      hand-crafted features    -> models/classical_ml.joblib
- CNN:               resized, normalised image -> models/cnn.pt

For each model it reports the raw probability of the malignant /
pre-malignant group and, if models/<name>_calibrator.json exists
(src/eval/recalibration.py), the recalibrated probability as well. Models
whose files are missing are skipped with a note rather than an error.

Usage:
    # any image file
    python -m src.predict path/to/image.jpg --age 62 --sex male --localization back

    # a HAM10000 image by ID: finds the file and its metadata, shows the true label
    python -m src.predict --image-id ISIC_0029176

Images from the held-out test set are refused when looked up by ID. That
set is evaluated once, by the final evaluation script, and must not be
inspected case by case before then.

Research and educational prototype. Not a medical device. Not for clinical use.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.eval.recalibration import apply_recalibration, load_calibrator
from src.features.handcrafted import FEATURE_NAMES, extract_features_from_path
from src.models.metadata_baseline import FEATURE_COLUMNS

MODEL_NAMES = ("metadata_baseline", "classical_ml", "cnn")
MODEL_FILES = {
    "metadata_baseline": "metadata_baseline.joblib",
    "classical_ml": "classical_ml.joblib",
    "cnn": "cnn.pt",
}
DEFAULT_CNN_IMAGE_SIZE = 96


class InvalidImageError(ValueError):
    """The input could not be read as an image."""


def check_image(image_path) -> Path:
    path = Path(image_path)
    if not path.is_file():
        raise InvalidImageError(f"no file at {path}")
    try:
        with Image.open(path) as im:
            im.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise InvalidImageError(f"{path.name} could not be read as an image") from exc
    return path


def _cnn_image_size(reports_dir: Path) -> int:
    metrics = Path(reports_dir) / "cnn" / "metrics.json"
    if metrics.exists():
        with open(metrics) as f:
            return int(json.load(f).get("image_size", DEFAULT_CNN_IMAGE_SIZE))
    return DEFAULT_CNN_IMAGE_SIZE


def load_models(models_dir=Path("models"), reports_dir=Path("reports")) -> dict:
    """Load every model whose file exists. Returns
    {"models": {name: object}, "calibrators": {name: dict}, "skipped": {name: reason}, "cnn_image_size": int}."""
    models_dir = Path(models_dir)
    loaded = {"models": {}, "calibrators": {}, "skipped": {}, "cnn_image_size": _cnn_image_size(reports_dir)}

    for name in MODEL_NAMES:
        path = models_dir / MODEL_FILES[name]
        if not path.exists():
            loaded["skipped"][name] = f"{path} not found"
            continue
        if name == "cnn":
            try:
                import torch

                from src.models.cnn import SimpleCNN
            except ImportError:
                loaded["skipped"][name] = "PyTorch is not installed"
                continue
            model = SimpleCNN()
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            loaded["models"][name] = model
        else:
            loaded["models"][name] = joblib.load(path)

        calibrator_path = models_dir / f"{name}_calibrator.json"
        if calibrator_path.exists():
            loaded["calibrators"][name] = load_calibrator(calibrator_path)

    return loaded


def _predict_metadata(model, age, sex, localization) -> float:
    row = pd.DataFrame(
        [{"age": float("nan") if age is None else float(age), "sex": sex, "localization": localization}]
    )[FEATURE_COLUMNS]
    return float(model.predict_proba(row)[0, 1])


def _predict_classical(model, image_path: Path) -> float:
    features = pd.DataFrame([extract_features_from_path(image_path)])[FEATURE_NAMES]
    return float(model.predict_proba(features)[0, 1])


def _predict_cnn(model, image_path: Path, image_size: int) -> float:
    import torch

    from src.models.cnn import get_transforms

    with Image.open(image_path) as im:
        tensor = get_transforms(image_size, train=False)(im.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        return float(torch.sigmoid(model(tensor))[0])


def predict_image(image_path, loaded: dict, age=None, sex=None, localization=None) -> dict:
    """Return {model_name: {"raw": float, "recalibrated": float | None}} for
    the loaded models. Raises InvalidImageError for unreadable input."""
    path = check_image(image_path)
    sex = sex.strip().lower() if isinstance(sex, str) and sex.strip() else None
    localization = localization.strip().lower() if isinstance(localization, str) and localization.strip() else None

    results = {}
    for name, model in loaded["models"].items():
        if name == "metadata_baseline":
            raw = _predict_metadata(model, age, sex, localization)
        elif name == "classical_ml":
            raw = _predict_classical(model, path)
        else:
            raw = _predict_cnn(model, path, loaded["cnn_image_size"])

        calibrator = loaded["calibrators"].get(name)
        recalibrated = float(apply_recalibration([raw], calibrator)[0]) if calibrator else None
        results[name] = {"raw": raw, "recalibrated": recalibrated}
    return results


def lookup_image_id(image_id: str, raw_dir: Path, splits_dirs) -> dict:
    """Find a HAM10000 image by ID. Returns its path, split, true label and
    metadata. Refuses held-out test images."""
    raw_dir = Path(raw_dir)
    found = None
    for splits_dir in splits_dirs:
        for csv in sorted(Path(splits_dir).glob("*.csv")):
            df = pd.read_csv(csv)
            if "image_id" not in df.columns or "split" not in df.columns:
                continue
            match = df[df["image_id"] == image_id]
            if len(match):
                found = match.iloc[0]
                break
        if found is not None:
            break
    if found is None:
        raise ValueError(f"{image_id} is not in any split manifest")
    if found["split"] == "test":
        raise PermissionError(
            f"{image_id} is in the held-out test set. That set is evaluated once by the final "
            "evaluation script; use a validation (tune) image instead."
        )

    image_path = raw_dir / "HAM10000_images" / f"{image_id}.jpg"
    info = {"path": image_path, "split": found["split"], "dx": found["dx"], "label": int(found["label"]),
            "age": None, "sex": None, "localization": None}

    metadata_path = raw_dir / "HAM10000_metadata.tab"
    if metadata_path.exists():
        meta = pd.read_csv(metadata_path, sep=None, engine="python")
        row = meta[meta["image_id"] == image_id]
        if len(row):
            row = row.iloc[0]
            info["age"] = None if pd.isna(row["age"]) else float(row["age"])
            info["sex"] = None if pd.isna(row["sex"]) else str(row["sex"])
            info["localization"] = None if pd.isna(row["localization"]) else str(row["localization"])
    return info


def format_results(results: dict, skipped: dict) -> str:
    lines = [f"{'model':<20}{'raw':>8}{'recalibrated':>15}"]
    for name in MODEL_NAMES:
        if name in results:
            r = results[name]
            recal = "-" if r["recalibrated"] is None else f"{100 * r['recalibrated']:.1f}%"
            lines.append(f"{name:<20}{100 * r['raw']:>7.1f}%{recal:>15}")
        elif name in skipped:
            lines.append(f"{name:<20}skipped: {skipped[name]}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", type=Path, help="path to an image file")
    parser.add_argument("--image-id", help="HAM10000 image ID, e.g. ISIC_0029176")
    parser.add_argument("--age", type=float, default=None)
    parser.add_argument("--sex", default=None, help="male / female")
    parser.add_argument("--localization", default=None, help="body site, e.g. back, face, lower extremity")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--splits-dir", type=Path, nargs="+", default=[Path("data/splits_v2"), Path("data/splits")])
    args = parser.parse_args()

    if (args.image is None) == (args.image_id is None):
        parser.error("give either an image path or --image-id")

    truth = None
    if args.image_id:
        try:
            info = lookup_image_id(args.image_id, args.raw_dir, args.splits_dir)
        except (ValueError, PermissionError) as exc:
            raise SystemExit(str(exc))
        image_path = info["path"]
        age = info["age"] if args.age is None else args.age
        sex = info["sex"] if args.sex is None else args.sex
        localization = info["localization"] if args.localization is None else args.localization
        truth = f"true diagnosis: {info['dx']} (label {info['label']}), split: {info['split']}"
    else:
        image_path, age, sex, localization = args.image, args.age, args.sex, args.localization

    loaded = load_models(args.models_dir, args.reports_dir)
    if not loaded["models"]:
        raise SystemExit(f"No trained models found in {args.models_dir}/. Train them first (see README).")

    try:
        results = predict_image(image_path, loaded, age=age, sex=sex, localization=localization)
    except InvalidImageError as exc:
        raise SystemExit(f"Invalid input: {exc}")

    print("Probability of malignant / pre-malignant group")
    print(format_results(results, loaded["skipped"]))
    if truth:
        print(truth)
    print("Research prototype. Not a medical device. Not for clinical use.")


if __name__ == "__main__":
    main()
