# Skin Lesion Triage: Comparing Models and Their Calibration

<<<<<<< HEAD
** EXSU 500: Fundamentals of AI in Medicine, McGill University, Fall 2026 **
=======
**EXSU 500: Fundamentals of AI in Medicine, McGill University, Fall 2026**
>>>>>>> modelling/mourad

## Problem

We classify dermoscopic images of skin lesions as **malignant or pre-malignant** (melanoma, basal cell carcinoma, actinic keratosis / intraepithelial carcinoma) or **benign** (melanocytic nevus, benign keratosis, dermatofibroma, vascular lesion). The intended use is to support triage of lesions for dermatology referral.

We compare three types of model:

1. A baseline using patient metadata only (age, sex, body site).
2. A classical machine learning model using hand-crafted image features.
3. A fine-tuned convolutional neural network (CNN).

The models are compared on two questions:

- **Discrimination:** how well does each model separate malignant from benign lesions? Measured with AUROC, AUPRC, and sensitivity/specificity at a stated threshold.
- **Calibration:** can its predicted probabilities be trusted? For example, do lesions given an "80% risk" turn out to be malignant about 80% of the time? Measured with reliability diagrams, Brier score, expected calibration error, and calibration slope and intercept, both before and after post-hoc recalibration.

**Why it matters:** skin cancer is the most common cancer, and outcomes depend on catching malignant lesions early. A triage tool is only useful if clinicians can act on its output. A model that ranks lesions well but reports overconfident probabilities can still push referral decisions in the wrong direction.

**Task type:** binary image classification (supervised), with calibration analysis.

## Dataset

**HAM10000** (Human Against Machine with 10,000 training images): 10,015 dermoscopic images of 7,470 unique pigmented skin lesions, with diagnosis, age, sex and body site.

- **Source:** Harvard Dataverse, [doi:10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T)
- **Licence:** CC BY-NC 4.0 (non-commercial use)
- **Citation:** Tschandl P, Rosendahl C, Kittler H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data* 5, 180161 (2018).

**Splits** (declared before any model was trained, at the lesion level so no lesion's images cross a split boundary; see `src/data/make_splits.py` and the committed manifests in `data/splits/`):

- **Train:** 8,488 images / 6,349 lesions.
- **Validation:** 1,527 images / 1,121 lesions. All tuning, thresholding and recalibration use this set only.
- **Held-out test:** the official ISIC 2018 Task 3 test set, 1,512 images, downloaded from the same Dataverse record. Verified to share zero `image_id`/`lesion_id` with train or validation. It is evaluated once, after all models are finalised.
- **Reference comparison:** reader and CNN data from Tschandl et al., *Nature Medicine* 26, 1229–1234 (2020), included in the same download. Used only for comparison, never for training.

The data are **not** committed to this repository (see `.gitignore`).

## Getting the data

```bash
python -m src.data.download --out data/raw   # downloads ~3.1 GB from Harvard Dataverse
# then unzip the .zip files it downloads, e.g.:
cd data/raw && for f in *.zip; do unzip -o "$f"; done
```

This produces `data/raw/HAM10000_images/` (10,015 `.jpg`), `data/raw/HAM10000_segmentations_lesion_tschandl/` (10,015 masks), `data/raw/ISIC2018_Task3_Test_Images/` (1,511 `.jpg`; the test ground-truth file lists one additional `image_id`, `ISIC_0035068`, that was withdrawn from this Dataverse distribution), and the `.tab` metadata/ground-truth files. `data/splits/*.csv` are already committed, so this step is only needed to retrain a model or rebuild the splits/features from scratch.

## Reproducing the results

```bash
pip install -r requirements.txt

# only needed if you don't already have data/features/{train,val}.csv committed:
python -m src.features.build_feature_table --splits train val

python -m src.models.metadata_baseline   # model 1: metadata-only baseline
python -m src.models.classical_ml        # model 2: hand-crafted image features

python -m pytest tests/ -q
```

Each model script trains on `data/splits/train.csv`, reports on `data/splits/val.csv` only (the test set is touched once, at the end, across all models), and writes its metrics/reliability diagram to `reports/<model>/`.

## Results

Validation-set performance (95% bootstrap CI, 1000 resamples), threshold 0.5:

| Model | AUROC | AUPRC | Sensitivity | Specificity | Brier score | ECE |
|---|---|---|---|---|---|---|
| 1. Metadata baseline (age/sex/site, logistic regression) | 0.787 [0.761, 0.813] | 0.424 [0.376, 0.481] | 0.800 [0.755, 0.845] | 0.688 [0.663, 0.712] | 0.199 | 0.241 |
| 2. Classical ML (hand-crafted features, gradient boosting) | 0.840 [0.818, 0.862] | 0.555 [0.502, 0.616] | 0.747 [0.697, 0.799] | 0.730 [0.708, 0.756] | 0.160 | 0.163 |
| 3. CNN (trained from scratch, no pretrained backbone) | 0.850 [0.830, 0.869] | 0.528 [0.472, 0.588] | 0.920 [0.889, 0.949] | 0.643 [0.617, 0.668] | 0.197 | 0.253 |

The classical model's AUROC CI doesn't overlap the baseline's, so the gain from adding image information is a real effect, not noise; the CNN's AUROC CI overlaps the classical model's, so that further gain is not clearly distinguishable from noise at this sample size. All three models are noticeably overconfident (reliability diagrams in `reports/*/reliability_diagram.png` sit below the diagonal), worsening from model 1 to model 3 — a known side effect of class-imbalance handling (`class_weight="balanced"` / a `pos_weight` loss) shifting predicted probabilities away from the true base rate, and the motivation for the post-hoc recalibration step mentioned above. The CNN was trained without a pretrained ImageNet backbone because this environment's network policy blocks both `download.pytorch.org` and `huggingface.co`; a transfer-learning model would likely perform better.

## Team roles

| Role | Member |
|---|---|
| Data lead | Giuliana Curcio |
| Modelling lead | Abdul Aziz Mourad |
| Interface lead | Sophie Lee |
| Writing lead | Ariel Subekti |

See [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md) for the CRediT contribution statement and Use of Generative AI statement, and [`MODEL_CARD.md`](MODEL_CARD.md) for the current best model's intended use, data, metrics, and limitations.

## Licence

Code: MIT (see `LICENSE`). Data: CC BY-NC 4.0, owned by the dataset authors and not redistributed here.

*Course prototype for research and educational purposes. Not a medical device and not for clinical use.*
