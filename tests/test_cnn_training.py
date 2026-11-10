"""CNN training script end to end on a handful of tiny synthetic images:
device selection, the three-way split options, saved predictions, recorded
settings, and loading the saved weights back for single-image prediction.
The Apple-GPU tests run only on a Mac with MPS available."""
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src import predict  # noqa: E402
from src.models import cnn  # noqa: E402

HAS_MPS = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
needs_mps = pytest.mark.skipif(not HAS_MPS, reason="Apple MPS device not available")


class PythonFloatLabels(Dataset):
    """Labels as Python floats, like HAM10000ImageDataset: the default
    collation turns them into float64 tensors, which MPS cannot hold."""

    def __init__(self, n, size=32):
        generator = torch.Generator().manual_seed(0)
        self.images = torch.randn(n, 3, size, size, generator=generator)
        self.labels = [float(i % 2) for i in range(n)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def test_pick_device():
    assert cnn.pick_device("cpu").type == "cpu"
    assert cnn.pick_device().type in {"cpu", "cuda", "mps"}
    if HAS_MPS and not torch.cuda.is_available():
        assert cnn.pick_device().type == "mps"


def test_train_model_accepts_python_float_labels_on_cpu():
    loader = DataLoader(PythonFloatLabels(16), batch_size=8)
    assert next(iter(loader))[1].dtype == torch.float64  # the situation being guarded against
    model, history, _ = cnn.train_model(
        cnn.SimpleCNN(), loader, loader, torch.device("cpu"), epochs=1, lr=1e-3, patience=5, pos_weight=np.float64(2.5)
    )
    assert len(history) == 1


@needs_mps
def test_train_model_runs_on_apple_gpu():
    device = torch.device("mps")
    loader = DataLoader(PythonFloatLabels(16), batch_size=8)
    model, history, _ = cnn.train_model(
        cnn.SimpleCNN().to(device), loader, loader, device, epochs=1, lr=1e-3, patience=5, pos_weight=np.float64(2.5)
    )
    labels, probs = cnn.predict_probs(model, loader, device)
    assert len(history) == 1 and probs.shape == (16,) and np.isfinite(probs).all()


def _tiny_dataset(tmp_path, sizes):
    """Synthetic images plus manifests. sizes: {split name: number of images}."""
    image_dir = tmp_path / "raw" / "HAM10000_images"
    image_dir.mkdir(parents=True)
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    rng = np.random.default_rng(0)
    counter = 0
    for split, n in sizes.items():
        rows = []
        for i in range(n):
            counter += 1
            image_id, label = f"ISIC_{counter:07d}", i % 2
            pixels = rng.integers(0, 120 if label else 255, size=(40, 40, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(image_dir / f"{image_id}.jpg")
            rows.append({"image_id": image_id, "lesion_id": f"L{counter}", "group_id": f"L{counter}",
                         "dx": "mel" if label else "nv", "label": label, "split": split})
        pd.DataFrame(rows).to_csv(splits_dir / f"{split}.csv", index=False)
    return splits_dir


def _run(monkeypatch, tmp_path, splits_dir, *extra):
    argv = ["cnn", "--splits-dir", str(splits_dir), "--raw-dir", str(tmp_path / "raw"),
            "--out", str(tmp_path / "models" / "cnn.pt"), "--reports-dir", str(tmp_path / "reports" / "cnn"),
            "--epochs", "1", "--batch-size", "8", "--image-size", "32", "--num-workers", "0", "--n-boot", "20",
            *extra]
    monkeypatch.setattr("sys.argv", argv)
    cnn.main()
    with open(tmp_path / "reports" / "cnn" / "metrics.json") as f:
        return json.load(f)


def test_three_way_split_run_end_to_end(tmp_path, monkeypatch):
    splits_dir = _tiny_dataset(tmp_path, {"train": 16, "tune": 8, "calibration": 8})
    report = _run(monkeypatch, tmp_path, splits_dir, "--device", "cpu",
                  "--eval-split", "tune", "--predict-splits", "tune", "calibration")

    assert report["eval_split"] == "tune" and report["n_train"] == 16 and report["n_val"] == 8
    settings = report["settings"]
    assert settings["device"] == "cpu" and settings["batch_size"] == 8 and settings["seed"] == 42
    assert settings["pos_weight"] == 1.0 and settings["max_epochs"] == 1

    for split in ("tune", "calibration"):
        predictions = pd.read_csv(tmp_path / "reports" / "cnn" / f"predictions_{split}.csv")
        manifest = pd.read_csv(splits_dir / f"{split}.csv")
        assert list(predictions["image_id"]) == list(manifest["image_id"])
        assert list(predictions.columns) == ["image_id", "lesion_id", "group_id", "split", "label", "prob"]
        assert predictions["prob"].between(0, 1).all()

    # the saved weights load back and give the same probability for the same image
    loaded = predict.load_models(tmp_path / "models", tmp_path / "reports")
    assert loaded["cnn_image_size"] == 32
    first = pd.read_csv(tmp_path / "reports" / "cnn" / "predictions_tune.csv").iloc[0]
    image = tmp_path / "raw" / "HAM10000_images" / f"{first['image_id']}.jpg"
    assert predict.predict_image(image, loaded)["cnn"]["raw"] == pytest.approx(first["prob"], abs=1e-5)


def test_default_run_uses_val_split(tmp_path, monkeypatch):
    splits_dir = _tiny_dataset(tmp_path, {"train": 16, "val": 8})
    report = _run(monkeypatch, tmp_path, splits_dir, "--device", "cpu")
    assert report["eval_split"] == "val"
    assert not list((tmp_path / "reports" / "cnn").glob("predictions_*.csv"))


@pytest.mark.parametrize("extra", [["--eval-split", "test"], ["--predict-splits", "calibration", "test"]])
def test_held_out_test_set_is_refused(tmp_path, monkeypatch, extra):
    splits_dir = _tiny_dataset(tmp_path, {"train": 16, "val": 8, "test": 8})
    with pytest.raises(SystemExit):
        _run(monkeypatch, tmp_path, splits_dir, "--device", "cpu", *extra)
    assert not (tmp_path / "models" / "cnn.pt").exists()


@needs_mps
def test_run_end_to_end_on_apple_gpu(tmp_path, monkeypatch):
    splits_dir = _tiny_dataset(tmp_path, {"train": 16, "tune": 8, "calibration": 8})
    report = _run(monkeypatch, tmp_path, splits_dir, "--device", "mps",
                  "--eval-split", "tune", "--predict-splits", "calibration")
    assert report["settings"]["device"] == "mps"
    predictions = pd.read_csv(tmp_path / "reports" / "cnn" / "predictions_calibration.csv")
    assert predictions["prob"].between(0, 1).all()
    loaded = predict.load_models(tmp_path / "models", tmp_path / "reports")  # weights were saved as CPU tensors
    assert "cnn" in loaded["models"]
