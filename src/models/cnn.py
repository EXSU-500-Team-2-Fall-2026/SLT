"""Model 3 of 3: a small convolutional network trained from scratch on raw
HAM10000 images (see README.md). No pretrained backbone: pretrained
ImageNet weights could not be downloaded in the environment where this
model was first trained, so a compact CNN with data augmentation was used
instead (~8.5k training images). Transfer learning is a possible later
comparison.

Trains on <splits-dir>/train.csv and uses one development split
(--eval-split, default val) both to choose the best epoch and to report
on. The held-out test set is refused here: it is scored once, by the
final evaluation, after everything is frozen.

The script uses a GPU when one is available (NVIDIA CUDA, or Apple
silicon through MPS) and the CPU otherwise; --device overrides this. The
settings of every run are written to metrics.json. Results are close but
not bit-identical across devices and library versions.

Usage:
    # original split (midterm results)
    python -m src.models.cnn

    # duplicate-aware split: select and report on tune, save predictions for the calibrator
    python -m src.models.cnn \\
        --splits-dir data/splits_v2 --eval-split tune --predict-splits tune calibration \\
        --reports-dir reports/v2/cnn
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.eval.metrics import discrimination_metrics, reliability_diagram
from src.eval.reporting import add_split_arguments, check_development_splits, evaluate, split_label, write_predictions

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class HAM10000ImageDataset(Dataset):
    def __init__(self, split_csv: Path, image_dir: Path, transform):
        self.image_dir = Path(image_dir)
        df = pd.read_csv(split_csv)
        df = df[[(self.image_dir / f"{iid}.jpg").exists() for iid in df["image_id"]]].reset_index(drop=True)
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.image_dir / f"{row.image_id}.jpg").convert("RGB")
        return self.transform(image), float(row.label)


def get_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
            block(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.head(x).squeeze(1)


def pick_device(name: str = "auto") -> torch.device:
    """CUDA if available, else Apple-silicon MPS, else CPU; or the named device."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for images, y in loader:
        logits = model(images.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def train_model(
    model, train_loader, val_loader, device, epochs: int, lr: float, patience: int, pos_weight: float
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # float32 throughout: Apple MPS has no float64, and the default DataLoader
    # collation turns Python float labels into float64 tensors.
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32, device=device)
    )

    best_auroc = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(dtype=torch.float32).to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_labels, val_probs = predict_probs(model, val_loader, device)
        val_auroc = discrimination_metrics(val_labels, val_probs).auroc
        elapsed = time.time() - epoch_start

        history.append({"epoch": epoch, "train_loss": train_loss, "val_auroc": val_auroc, "seconds": elapsed})
        print(f"epoch {epoch:2d}  train_loss={train_loss:.4f}  val_auroc={val_auroc:.4f}  ({elapsed:.1f}s)")

        improved = best_state is None or (not np.isnan(val_auroc) and val_auroc > best_auroc)
        if improved:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"early stopping at epoch {epoch} (no val AUROC improvement for {patience} epochs)")
                break

    model.load_state_dict(best_state)
    return model, history, best_auroc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("models/cnn.pt"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/cnn"))
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    add_split_arguments(parser)
    args = parser.parse_args()
    check_development_splits(parser, args.eval_split, *args.predict_splits)

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}")

    image_dir = args.raw_dir / "HAM10000_images"
    eval_transform = get_transforms(args.image_size, train=False)
    train_ds = HAM10000ImageDataset(
        args.splits_dir / "train.csv", image_dir, get_transforms(args.image_size, train=True)
    )
    eval_ds = HAM10000ImageDataset(args.splits_dir / f"{args.eval_split}.csv", image_dir, eval_transform)
    if len(train_ds) == 0 or len(eval_ds) == 0:
        raise SystemExit(f"No images found in {image_dir}. See 'Getting the data' in README.md.")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    n_pos = int(train_ds.df["label"].sum())
    n_neg = len(train_ds.df) - n_pos
    pos_weight = n_neg / n_pos

    model = SimpleCNN().to(device)
    start_time = time.time()
    model, history, best_eval_auroc = train_model(
        model, train_loader, eval_loader, device, args.epochs, args.lr, args.patience, pos_weight
    )
    training_seconds = time.time() - start_time

    eval_labels, eval_probs = predict_probs(model, eval_loader, device)
    sections, calib = evaluate(eval_labels, eval_probs, threshold=args.threshold, n_boot=args.n_boot)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, args.out)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    reliability_diagram(
        calib, f"CNN (trained from scratch) – {split_label(args.eval_split)}",
        args.reports_dir / "reliability_diagram.png",
    )

    for split in args.predict_splits:
        if split == args.eval_split:
            split_ds, split_probs = eval_ds, eval_probs
        else:
            split_ds = HAM10000ImageDataset(args.splits_dir / f"{split}.csv", image_dir, eval_transform)
            split_loader = DataLoader(
                split_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
            )
            _, split_probs = predict_probs(model, split_loader, device)
        write_predictions(split_ds.df, split_probs, split, args.reports_dir / f"predictions_{split}.csv")

    report = {
        "model": "cnn",
        "architecture": "SimpleCNN (4 conv blocks, trained from scratch, no pretrained weights)",
        "image_size": args.image_size,
        "n_train": len(train_ds),
        "n_val": len(eval_ds),
        "splits_dir": args.splits_dir.as_posix(),
        "eval_split": args.eval_split,
        "settings": {
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_epochs": args.epochs,
            "patience": args.patience,
            "seed": args.seed,
            "pos_weight": float(pos_weight),
            "checkpoint_selection": f"highest AUROC on the {args.eval_split} split",
            "device": str(device),
            "torch_version": torch.__version__,
            "training_seconds": round(training_seconds, 1),
        },
        "epochs_run": len(history),
        "history": history,
        **sections,
    }
    with open(args.reports_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))

if __name__ == "__main__":
    main()
