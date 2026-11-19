"""Grad-CAM heat maps for the CNN (model 3).

Grad-CAM (Selvaraju et al., ICCV 2017) shows which regions of an image
raised the model's score. Here the score is the logit of the
malignant/premalignant group, and the layer is the output of the last
convolutional block before its pooling step (12 x 12 positions for a
96 x 96 input). Each of the 128 feature maps is weighted by the average
gradient of the score with respect to that map; the weighted maps are
summed, negative values are set to zero, and the result is enlarged to the
image size and scaled so that its maximum is 1.

How to read a map, and how not to:
- Warm regions are where the features that pushed the score *up* were
  active. A map that is empty means no region pushed the score up; it does
  not mean the model looked at nothing.
- Every map is scaled to its own maximum. Colours show where, relative to
  the rest of the same image; they cannot be compared between images and
  say nothing about how high the probability is.
- The resolution is coarse (each position covers 8 x 8 pixels of the
  network input), so a map cannot outline a lesion border.
- A map describes the model, not the lesion. It is not a segmentation, and
  a plausible-looking map is not evidence that a prediction is correct.

Usage:
    # one image -> original and overlay side by side
    python -m src.explain.gradcam --image-id ISIC_0025234 --out reports/v2/gradcam/ISIC_0025234.png
    python -m src.explain.gradcam path/to/image.jpg --out heatmap.png

    # a panel of tuning-subset examples, sampled at random with a fixed seed
    # within each outcome group (true/false positives and negatives)
    python -m src.explain.gradcam --panel --out reports/v2/gradcam/panel_tune.png

Images from the held-out test set are refused when looked up by ID, and
the panel draws on development predictions only.

Research and educational prototype. Not a medical device. Not for clinical use.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.eval.reporting import HELD_OUT_SPLIT

OUTCOME_GROUPS = ("true positive", "false negative", "false positive", "true negative")


def grad_cam(model, image_tensor) -> tuple[np.ndarray, float]:
    """Heat map for one normalised image tensor of shape (3, H, W) or
    (1, 3, H, W). Returns (map of shape (H, W) with values in [0, 1], the
    model's raw probability for the same forward pass). The map is all
    zeros when no region raised the score."""
    import torch
    import torch.nn.functional as F

    x = image_tensor.unsqueeze(0) if image_tensor.dim() == 3 else image_tensor
    if x.shape[0] != 1:
        raise ValueError("grad_cam explains one image at a time")
    device = next(model.parameters()).device
    x = x.to(device).clone().requires_grad_(True)  # keeps a graph even if the weights are frozen

    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            blocks = list(model.features.children())
            hidden = x
            for block in blocks[:-1]:
                hidden = block(hidden)
            *convolution_part, pooling = list(blocks[-1].children())
            for layer in convolution_part:
                hidden = layer(hidden)
            activations = hidden  # (1, channels, h, w): last block, before pooling
            logit = model.head(model.pool(pooling(activations))).squeeze(1)
            gradients = torch.autograd.grad(logit.sum(), activations)[0]

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach()
        peak = cam.max()
        if peak > 0:
            cam = cam / peak
        probability = float(torch.sigmoid(logit.detach())[0])
    finally:
        model.train(was_training)

    return cam.clamp(0, 1).cpu().numpy().astype(np.float32), probability


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.55) -> Image.Image:
    """Blend the heat map over the image at the image's own size. The tint
    grows with the map value, so regions with a value of zero are left
    exactly as they were."""
    import matplotlib

    image = image.convert("RGB")
    heat = Image.fromarray(np.uint8(np.clip(cam, 0, 1) * 255)).resize(image.size, Image.BILINEAR)
    strength = np.asarray(heat, dtype=np.float32) / 255.0
    colours = matplotlib.colormaps["jet"](strength)[..., :3] * 255.0
    weight = (alpha * strength)[..., None]
    blended = (1 - weight) * np.asarray(image, dtype=np.float32) + weight * colours
    return Image.fromarray(np.uint8(np.clip(np.rint(blended), 0, 255)))


def explain_image(image_path, model, image_size: int) -> tuple[Image.Image, Image.Image, float, np.ndarray]:
    """Returns (original image, overlay, raw probability, heat map at the
    network input size) using the same preprocessing as training."""
    from src.models.cnn import get_transforms

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    cam, probability = grad_cam(model, get_transforms(image_size, train=False)(image))
    return image, overlay_heatmap(image, cam), probability, cam


def select_examples(predictions: pd.DataFrame, threshold: float, per_group: int = 2, seed: int = 42) -> pd.DataFrame:
    """Sample up to `per_group` images at random, with a fixed seed, from
    each outcome group at the given threshold on the raw probability.
    Random sampling within groups, not picking the most striking cases,
    keeps the panel representative. Rows keep their predictions and gain an
    'outcome' column."""
    if "split" in predictions.columns and (predictions["split"] == HELD_OUT_SPLIT).any():
        raise ValueError("the example panel must not be built from held-out test predictions")
    flagged = predictions["prob"] >= threshold
    positive = predictions["label"] == 1
    outcome = np.select(
        [flagged & positive, ~flagged & positive, flagged & ~positive],
        list(OUTCOME_GROUPS[:3]),
        default=OUTCOME_GROUPS[3],
    )
    labelled = predictions.assign(outcome=outcome)
    chosen = []
    for group in OUTCOME_GROUPS:
        members = labelled[labelled["outcome"] == group]
        chosen.append(members.sample(n=min(per_group, len(members)), random_state=seed))
    return pd.concat(chosen, ignore_index=True)


def save_side_by_side(image: Image.Image, overlay: Image.Image, title: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, picture, name in zip(axes, (image, overlay), ("Image", "Grad-CAM")):
        ax.imshow(picture)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_panel(rows: list[dict], out_path: Path) -> None:
    """rows: dicts with image, overlay, outcome, dx, probability."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.6), squeeze=False)
    for column, row in enumerate(rows):
        axes[0, column].imshow(row["image"])
        axes[0, column].set_title(f"{row['outcome']}\n{row['dx']}, p = {row['probability']:.2f}", fontsize=8)
        axes[1, column].imshow(row["overlay"])
        for ax in axes[:, column]:
            ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _load_cnn(models_dir: Path, reports_dir: Path):
    from src.predict import load_models

    loaded = load_models(models_dir, reports_dir)
    if "cnn" not in loaded["models"]:
        raise SystemExit(f"CNN not available: {loaded['skipped'].get('cnn', 'unknown reason')}")
    return loaded["models"]["cnn"], loaded["cnn_image_size"]


def main() -> None:
    from src.predict import InvalidImageError, check_image, lookup_image_id

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", type=Path, help="path to an image file")
    parser.add_argument("--image-id", help="HAM10000 image ID, e.g. ISIC_0025234")
    parser.add_argument("--panel", action="store_true", help="panel of sampled tuning-subset examples")
    parser.add_argument("--out", type=Path, required=True, help="output .png file")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/v2"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--splits-dir", type=Path, nargs="+", default=[Path("data/splits_v2"), Path("data/splits")])
    parser.add_argument("--predictions", type=Path, default=None,
                        help="predictions file for --panel (default: <reports-dir>/cnn/predictions_tune.csv)")
    parser.add_argument("--threshold", type=float, default=0.5, help="raw-probability threshold for --panel groups")
    parser.add_argument("--per-group", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if sum([args.image is not None, args.image_id is not None, args.panel]) != 1:
        parser.error("give exactly one of: an image path, --image-id, or --panel")

    model, image_size = _load_cnn(args.models_dir, args.reports_dir)

    if args.panel:
        predictions_path = args.predictions or args.reports_dir / "cnn" / "predictions_tune.csv"
        chosen = select_examples(pd.read_csv(predictions_path), args.threshold, args.per_group, args.seed)
        rows = []
        for record in chosen.to_dict("records"):
            info = lookup_image_id(record["image_id"], args.raw_dir, args.splits_dir)
            image, overlay, probability, _ = explain_image(info["path"], model, image_size)
            rows.append({"image": image, "overlay": overlay, "outcome": record["outcome"],
                         "dx": info["dx"], "probability": probability})
            print(f"{record['image_id']}  {record['outcome']:<15} dx={info['dx']:<6} p={probability:.3f}")
        save_panel(rows, args.out)
    else:
        if args.image_id:
            try:
                info = lookup_image_id(args.image_id, args.raw_dir, args.splits_dir)
            except (ValueError, PermissionError) as exc:
                raise SystemExit(str(exc))
            image_path = info["path"]
            title = f"{args.image_id}: true diagnosis {info['dx']} (split: {info['split']})"
        else:
            image_path, title = args.image, args.image.name
        try:
            check_image(image_path)
        except InvalidImageError as exc:
            raise SystemExit(f"Invalid input: {exc}")
        image, overlay, probability, cam = explain_image(image_path, model, image_size)
        save_side_by_side(image, overlay, f"{title}\nraw CNN probability {probability:.2f}", args.out)
        print(f"raw CNN probability {probability:.3f}; share of the image with map value above 0.5: "
              f"{float((cam > 0.5).mean()):.2f}")

    print(f"wrote {args.out}")
    print("Research prototype. Not a medical device. Not for clinical use.")


if __name__ == "__main__":
    main()
