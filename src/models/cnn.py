"""Model 3 of 3: a small convolutional network trained from scratch on raw
HAM10000 images (see README.md). No pretrained backbone -- this
environment's network policy blocks both download.pytorch.org and
huggingface.co, so ImageNet weights aren't reachable here. A compact CNN
with data augmentation is a reasonable, literature-precedented fallback at
this dataset size (~8.5k training images).

Trains on data/splits/train.csv, reports on data/splits/val.csv only; the
held-out test set is touched once, after all three models are built and
compared.

Usage:
    python -m src.models.cnn \\
        --splits-dir data/splits --raw-dir data/raw \\
        --out models/cnn.pt --reports-dir reports/cnn \\
        --epochs 20 --batch-size 64 --image-size 112
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

from src.eval.metrics import bootstrap_metrics_ci, calibration_metrics, discrimination_metrics, reliability_diagram

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
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    best_auroc = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
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
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_dir = args.raw_dir / "HAM10000_images"
    train_ds = HAM10000ImageDataset(
        args.splits_dir / "train.csv", image_dir, get_transforms(args.image_size, train=True)
    )
    val_ds = HAM10000ImageDataset(
        args.splits_dir / "val.csv", image_dir, get_transforms(args.image_size, train=False)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    n_pos = train_ds.df["label"].sum()
    n_neg = len(train_ds.df) - n_pos
    pos_weight = n_neg / n_pos

    model = SimpleCNN().to(device)
    model, history, best_val_auroc = train_model(
        model, train_loader, val_loader, device, args.epochs, args.lr, args.patience, pos_weight
    )

    val_labels, val_probs = predict_probs(model, val_loader, device)
    disc = discrimination_metrics(val_labels, val_probs, threshold=args.threshold)
    calib = calibration_metrics(val_labels, val_probs)
    ci = bootstrap_metrics_ci(val_labels, val_probs, threshold=args.threshold)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    reliability_diagram(
        calib, "CNN (trained from scratch) – validation", args.reports_dir / "reliability_diagram.png"
    )

    report = {
        "model": "cnn",
        "architecture": "SimpleCNN (4 conv blocks, trained from scratch, no pretrained weights)",
        "image_size": args.image_size,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "epochs_run": len(history),
        "history": history,
        "discrimination": {
            "auroc": disc.auroc,
            "auroc_ci95": ci["auroc_ci"],
            "auprc": disc.auprc,
            "auprc_ci95": ci["auprc_ci"],
            "sensitivity": disc.sensitivity,
            "sensitivity_ci95": ci["sensitivity_ci"],
            "specificity": disc.specificity,
            "specificity_ci95": ci["specificity_ci"],
            "threshold": disc.threshold,
        },
        "calibration": {
            "brier_score": calib.brier_score,
            "brier_score_ci95": ci["brier_score_ci"],
            "ece": calib.ece,
            "calibration_slope": calib.calibration_slope,
            "calibration_intercept": calib.calibration_intercept,
        },
        "bootstrap": {"n_boot": ci["n_boot"], "ci_level": ci["ci_level"]},
    }
    with open(args.reports_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
