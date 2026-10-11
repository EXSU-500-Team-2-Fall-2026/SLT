# Model Card

**Status:** draft. All three models are now built; this card covers the
one with the highest validation AUROC (the primary metric our bootstrap
CIs treat as such), `cnn`. That said, the choice isn't clear-cut — see
"Model selection is a trade-off" below — and will be revisited once the
held-out test set is evaluated (once, after model selection is final) and
once post-hoc recalibration is implemented.

## Model details

- **Name:** `cnn` (`src/models/cnn.py`)
- **Type:** a small 4-block convolutional network (`SimpleCNN`), trained
  from scratch on raw images (96x96) with data augmentation. No
  pretrained backbone: this environment's network policy blocks both
  `download.pytorch.org` and `huggingface.co`.
- **Version:** trained from `data/splits/{train,val}.csv` and
  `data/raw/HAM10000_images/`, commit `<pending>` on branch
  `modelling/mourad`.

## Model selection is a trade-off, not a clear winner

| Metric | 1. Metadata baseline | 2. Classical ML | 3. CNN |
|---|---|---|---|
| AUROC | 0.787 [0.761, 0.813] | 0.840 [0.818, 0.862] | **0.850** [0.830, 0.869] |
| AUPRC | 0.424 | **0.555** | 0.528 |
| Sensitivity | 0.800 | 0.747 | **0.920** |
| Specificity | 0.688 | **0.730** | 0.643 |
| ECE (lower is better) | 0.241 | **0.163** | 0.253 |

The CNN's AUROC edge over the classical model is **not statistically
distinguishable** at this sample size (their 95% CIs overlap). The CNN
does have much higher sensitivity (0.92 vs. 0.75) — relevant given the
project's triage motivation, where missing a malignant lesion is the
costlier error — but at the price of more false positives (lower
specificity), lower AUPRC, and the worst calibration of the three models.
Which model is actually "best" depends on which of these the team
prioritizes; this card defaults to ranking by AUROC alone, which is a
simplification worth revisiting in the manuscript.

## Intended use

Research/educational prototype for the EXSU 500 course project. Intended
to illustrate and compare triage-support approaches for flagging
dermoscopic images of skin lesions as malignant/pre-malignant vs. benign,
**not** to make or support real clinical decisions.

**This is not a medical device and must not be used for clinical care.**

## Training data

HAM10000 training split: 8,488 images of 6,349 lesions (lesion-level
split, see `data/splits/train.csv`), sourced from the Harvard Dataverse
record [doi:10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T).
Labels are binarized from the 7-class `dx` diagnosis (malignant/
pre-malignant: melanoma, basal cell carcinoma, actinic keratosis;
benign: nevus, benign keratosis, dermatofibroma, vascular lesion; see
`src/data/labels.py`). The CNN trains on raw pixels directly (resized to
96x96, augmented); no hand-crafted features or segmentation are involved
for this model specifically (see `reports/classical_ml/` and its own
section of this limitations list for model 2's feature pipeline).

## Evaluation data

HAM10000 validation split: 1,527 images of 1,121 lesions
(`data/splits/val.csv`), disjoint from training at the lesion level. The
official ISIC 2018 Task 3 test set (1,512 images) is held out and has not
been used for any decision about this model yet.

## Metrics (validation set, 95% bootstrap CI, threshold 0.5)

| Metric | Value |
|---|---|
| AUROC | 0.850 [0.830, 0.869] |
| AUPRC | 0.528 [0.472, 0.588] |
| Sensitivity | 0.920 [0.889, 0.949] |
| Specificity | 0.643 [0.617, 0.668] |
| Brier score | 0.197 [0.186, 0.208] (calibration not yet recalibrated) |
| Expected calibration error | 0.253 |

Full metrics are in `reports/cnn/metrics.json`; the classical model's are
in `reports/classical_ml/metrics.json` for comparison.

## Known limitations

- **Calibration is poor out of the box, worse than either simpler
  model.** The reliability diagram (`reports/cnn/reliability_diagram.png`)
  shows systematic overconfidence: predicted probabilities run well above
  the observed malignancy rate. This traces to the `pos_weight` term in
  the training loss shifting predictions away from the true base rate.
  Post-hoc recalibration (planned, not yet implemented) is needed before
  any probability output should be trusted at face value.
- **No pretrained backbone.** Trained entirely from scratch on ~8,500
  images because this environment's network policy blocks ImageNet
  weight downloads. A transfer-learning model would likely generalize
  better; this result probably understates what a CNN-based approach can
  achieve on this task.
- **No patient identifier in the source data.** HAM10000's metadata gives
  `lesion_id` but no separate patient ID, so the lesion-level train/val
  split (the finest grouping the released data actually supports) cannot
  rule out the same patient contributing lesions to both splits.
- **Dataset provenance and population.** HAM10000 draws from a limited
  set of academic dermatology clinics in Austria and Australia. Skin tone
  distribution, imaging equipment, and patient demographics are not
  representative of the general population; performance on populations
  or imaging conditions outside this distribution is unknown and not
  validated here.
- **Threshold is not clinically derived.** 0.5 is the default operating
  point for reporting; it has not been chosen to optimise any specific
  clinical trade-off (e.g. minimum acceptable sensitivity for a referral
  pathway).
- **Not yet evaluated against the held-out test set**, so these numbers
  may not generalise; they are a validation-set estimate only.

## Populations this model should not be used on

Any real patient, in any clinical or non-clinical setting. As a research
prototype it has not been validated on populations outside HAM10000's
source clinics, on skin tones underrepresented in that dataset, on
pediatric patients, or on images taken with equipment/protocols
different from HAM10000's.
