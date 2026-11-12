# Model Card

**Status:** draft. This card describes the CNN, which has the highest observed AUROC on the tuning subset among the three implemented models. It does not establish a definitive best model. Selection criteria, models, preprocessing, recalibration, and thresholds will be finalized using development data before held-out evaluation. Test outcomes will be reported without revising these choices. A logistic recalibration mapping has been fitted for this model on the calibration subset; its independent held-out assessment is planned, not done.

## Model details

- **Name:** `cnn` (`src/models/cnn.py`)
- **Type:** a small 4-block convolutional network (`SimpleCNN`), trained
  from scratch on raw images (96x96) with data augmentation. No
  pretrained backbone: pretrained weights could not be downloaded in the
  original training environment.
- **Version:** trained from `data/splits_v2/train.csv` (duplicate-aware
  split) and `data/raw/HAM10000_images/`, with the checkpoint selected on
  `data/splits_v2/tune.csv`. Results and run settings are in
  `reports/v2/cnn/`, committed in `475a71b` and merged into `main` on
  2026-11-12. An earlier preliminary version trained on the original
  split is in `reports/cnn/` (commit `a8635b8`).

## Model comparison on the tuning subset

Development performance on 752 images of 560 lesions, before recalibration, threshold 0.5.

| Metric | 1. Metadata baseline | 2. Classical ML | 3. CNN |
|---|---|---|---|
| AUROC | 0.803 [0.765, 0.841] | 0.845 [0.814, 0.874] | **0.872** [0.848, 0.897] |
| AUPRC | 0.455 | 0.510 | **0.537** |
| Sensitivity | 0.787 | 0.757 | **0.926** |
| Specificity | 0.698 | **0.752** | 0.680 |
| Brier score (lower is better) | 0.190 | **0.160** | 0.172 |
| ECE (lower is better) | 0.244 | **0.175** | 0.216 |

At the current threshold of 0.5, the CNN has higher observed sensitivity and lower specificity than histogram gradient boosting. It has higher AUROC and AUPRC, and also higher Brier score and ECE. These descriptive differences do not establish statistical superiority or equivalence. Overlap between separate CIs is not a test of model differences; paired lesion-group bootstrap comparisons are planned.

The models change both inputs and modelling approach, so the comparison does not isolate the effect of learned representations or complexity. The threshold is not clinically validated. The card's coverage of the CNN reflects its observed AUROC on the tuning subset, not a demonstrated clinical advantage.

## Intended use

Research/educational prototype for a retrospective comparison of discrimination and probability calibration in HAM10000. The endpoint groups malignant/premalignant diagnoses versus benign diagnoses; “higher-concern” and “lower-concern” are shorthand for those labels, not measured referral needs. The grouping does not imply identical severity or management within either group.

The motivation is lesion assessment and potential triage support. This experiment does not establish referral benefit, clinical outcomes, general population screening performance, or suitability for ordinary smartphone photographs.

**This is not a medical device and must not be used for clinical care.**

## Training data

HAM10000 training subset of the duplicate-aware split: 8,519 images of
6,350 lesions (`data/splits_v2/train.csv`; images grouped by lesion and by
the 18 known near-duplicate pairs), sourced from the Harvard Dataverse
record [doi:10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T).
Labels are binarized from the 7-class `dx` diagnosis (malignant/
pre-malignant: melanoma, basal cell carcinoma, actinic keratosis / intraepithelial carcinoma;
benign: nevus, benign keratosis, dermatofibroma, vascular lesion; see
`src/data/labels.py`). The CNN trains on raw pixels directly (resized to
96x96, augmented); no hand-crafted features or segmentation are involved
for this model specifically (see `reports/classical_ml/` and its own
feature pipeline).

## Evaluation data

HAM10000 tuning subset: 752 images of 560 lesions
(`data/splits_v2/tune.csv`), disjoint from training at the level of connected lesion/near-duplicate groups. The split uses grouped random sampling with seed 42, allocating 85% / 7.5% / 7.5% of groups to training, tuning, and calibration, without explicit diagnosis stratification. The tuning subset supports development evaluation and CNN checkpoint selection. A separate calibration subset (744 images of 560 lesions, `data/splits_v2/calibration.csv`) is used only to fit the recalibration mapping. The
official ISIC 2018 Task 3 test set (1,511 available images of the 1,512
listed; `ISIC_0035068` was withdrawn) is held out and has not
been used for any decision about this model, as documented by the team; the training and calibrator-fitting scripts refuse it. The manifest contains 1,223 lesions and the repository documents 1,222 among available images. Final evaluation code and checks for provenance, duplicates, label compatibility, metadata availability, preprocessing, exclusions, and comparable cohorts remain planned.

## Metrics (development results on the tuning subset, threshold 0.5)

These values were obtained on the duplicate-aware split, before recalibration. The checkpoint was selected on the same tuning subset, so they are optimistic to an unmeasured degree. They are not held-out results.

The listed 95% percentile CIs use 1,000 image-level bootstrap resamples. They do not account for within-lesion dependence, repeated training, or model selection on the tuning subset. Lesion-group CIs and paired comparisons are planned. CIs are currently computed for AUROC, AUPRC (average precision), sensitivity, specificity, and Brier score, not for ECE or calibration slope/intercept.

| Metric | Value |
|---|---|
| AUROC | 0.872 [0.848, 0.897] |
| AUPRC | 0.537 [0.458, 0.628] |
| Sensitivity | 0.926 [0.882, 0.969] |
| Specificity | 0.680 [0.645, 0.718] |
| Brier score | 0.172 [0.157, 0.186] (before recalibration) |
| Expected calibration error | 0.216 |
| Calibration slope / intercept | 1.19 / −1.78 |

Full metrics and run settings are in `reports/v2/cnn/metrics.json`; the other models' are in `reports/v2/metadata_baseline/` and `reports/v2/classical_ml/`. Preliminary results on the original split (validation AUROC 0.850 [0.830, 0.869]) remain in `reports/cnn/metrics.json`; the two sets use different evaluation images and are not directly comparable.

## Known limitations

- **Near-duplicate images.** The [data-branch split audit](https://github.com/EXSU-500-Team-2-Fall-2026/SLT/blob/efadcbabcc96c43b1a30512c3c00e0504bfbaabf/reports/dataset/split_audit.md) identified 5 of 18 known near-duplicate image pairs crossing the training/validation boundary of the original split despite having different lesion IDs. The duplicate-aware split keeps each of the 18 pairs within one subset, and this model was retrained on it. Only the published pairs are handled; other visual duplicates may exist, and zero shared image or lesion IDs does not establish independence of image content. The held-out cohort's overlap/eligibility checks are still required.

- **Probability reliability remains limited.** On the tuning subset the CNN's calibration intercept (−1.78) and slope (1.19), together with the reliability diagram, suggest overestimation of positive-group probabilities, and its Brier score and ECE are higher than those of the classical model. Class weighting is a plausible contributor, but this comparison does not isolate its effect. An unweighted logistic recalibration mapping has been fitted on the calibration subset, whose cases played no part in model fitting or checkpoint selection (intercept −1.408, slope 0.976 on the logit scale; `reports/v2/cnn/calibrator.json`). Whether it improves calibration is not known: original and recalibrated held-out predictions will be compared using Brier-score change and supporting calibration measures, with paired lesion-group uncertainty. See the [agreed protocol](reports/experimental-design.md#future-recalibration-protocol). No improvement is assumed and test outcomes will not select or refit calibrators.
- **No pretrained backbone.** The CNN was trained from scratch because pretrained weights could not be downloaded in the original environment. Transfer learning is a possible future comparison; a performance improvement has not been established here.


- **No patient identifier in the source data.** HAM10000's metadata gives
  `lesion_id` but no separate patient ID, so the grouped split cannot
  rule out the same patient contributing lesions to more than one subset.
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
- **Diagnosis-specific performance remains planned.** The combined endpoint can conceal errors for individual diagnoses; sensitivity by positive diagnosis, especially melanoma, will be assessed.
- **Not yet evaluated against the held-out test set**, so these numbers
  may not generalise; they are a tuning-subset estimate only.

## Populations this model should not be used on

Any real patient, in any clinical or non-clinical setting. As a research
prototype it has not been validated on populations outside HAM10000's
source clinics, on skin tones underrepresented in that dataset, on
pediatric patients, or on images taken with equipment/protocols
different from HAM10000's.
