# Model Card

**Status:** draft. This card describes the CNN, which has the highest observed validation AUROC among the three implemented models. It does not establish a definitive best model. Selection criteria, models, preprocessing, any recalibration, and thresholds will be finalized using development data before held-out evaluation. Test outcomes will be reported without revising these choices. Post-hoc recalibration remains planned.

## Model details

- **Name:** `cnn` (`src/models/cnn.py`)
- **Type:** a small 4-block convolutional network (`SimpleCNN`), trained
  from scratch on raw images (96x96) with data augmentation. No
  pretrained backbone: this environment's network policy blocks both
  `download.pytorch.org` and `huggingface.co`.
- **Version:** trained from `data/splits/{train,val}.csv` and
  `data/raw/HAM10000_images/`; results committed in `a8635b8`
  ("Real model 3/3 results: CNN trained from scratch") on branch
  `modelling/mourad`, merged into `main` on 2026-10-10.

## Preliminary model comparison

| Metric | 1. Metadata baseline | 2. Classical ML | 3. CNN |
|---|---|---|---|
| AUROC | 0.787 [0.761, 0.813] | 0.840 [0.818, 0.862] | **0.850** [0.830, 0.869] |
| AUPRC | 0.424 | **0.555** | 0.528 |
| Sensitivity | 0.800 | 0.747 | **0.920** |
| Specificity | 0.688 | **0.730** | 0.643 |
| ECE (lower is better) | 0.241 | **0.163** | 0.253 |

At the current threshold of 0.5, the CNN has higher observed sensitivity and lower specificity than histogram gradient boosting. It also has lower AUPRC and higher ECE. These descriptive differences do not establish statistical superiority or equivalence. Overlap between separate CIs is not a test of model differences; paired lesion-level bootstrap comparisons are planned.

The models change both inputs and modelling approach, so the comparison does not isolate the effect of learned representations or complexity. The threshold is not clinically validated. The card's coverage of the CNN reflects its observed validation AUROC, not a demonstrated clinical advantage.

## Intended use

Research/educational prototype for a retrospective comparison of discrimination and probability calibration in HAM10000. The endpoint groups malignant/premalignant diagnoses versus benign diagnoses; “higher-concern” and “lower-concern” are shorthand for those labels, not measured referral needs. The grouping does not imply identical severity or management within either group.

The motivation is lesion assessment and potential triage support. This experiment does not establish referral benefit, clinical outcomes, general population screening performance, or suitability for ordinary smartphone photographs.

**This is not a medical device and must not be used for clinical care.**

## Training data

HAM10000 training split: 8,488 images of 6,349 lesions (lesion-level
split, see `data/splits/train.csv`), sourced from the Harvard Dataverse
record [doi:10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T).
Labels are binarized from the 7-class `dx` diagnosis (malignant/
pre-malignant: melanoma, basal cell carcinoma, actinic keratosis / intraepithelial carcinoma;
benign: nevus, benign keratosis, dermatofibroma, vascular lesion; see
`src/data/labels.py`). The CNN trains on raw pixels directly (resized to
96x96, augmented); no hand-crafted features or segmentation are involved
for this model specifically (see `reports/classical_ml/` and its own
feature pipeline).

## Evaluation data

HAM10000 validation split: 1,527 images of 1,121 lesions
(`data/splits/val.csv`), disjoint from training at the lesion level. The split uses grouped random sampling with seed 42 and 15% of lesion groups in validation, without explicit diagnosis stratification. Validation supports development evaluation and CNN early stopping. The
official ISIC 2018 Task 3 test set (1,511 available images of the 1,512
listed; `ISIC_0035068` was withdrawn) is held out and has not
been used for any decision about this model yet, as documented by the team. The manifest contains 1,223 lesions and the repository documents 1,222 among available images. Final evaluation code and checks for provenance, duplicates, label compatibility, metadata availability, preprocessing, exclusions, and comparable cohorts remain planned.

## Metrics (preliminary validation results, threshold 0.5)

These values were obtained using the original split, which has five known near-duplicate pairs crossing training and validation. They remain unchanged for the midterm and may be optimistic; no corrected experiment has been completed.

The listed 95% percentile CIs use 1,000 image-level bootstrap resamples. They do not account for within-lesion dependence, repeated training, or model selection on validation. Lesion-level CIs and paired comparisons are planned. CIs are currently computed for AUROC, AUPRC (average precision), sensitivity, specificity, and Brier score, not for ECE or calibration slope/intercept.

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

- **Known cross-split near-duplicates.** The [data-branch split audit](https://github.com/EXSU-500-Team-2-Fall-2026/SLT/blob/efadcbabcc96c43b1a30512c3c00e0504bfbaabf/reports/dataset/split_audit.md) identifies 5 of 18 known near-duplicate image pairs crossing the current training/validation boundary despite having different lesion IDs. Zero shared image IDs or lesion IDs therefore does not establish independence of image content. These pairs provide a potential information-leakage route; the magnitude and direction of their effect on individual metrics have not been measured. Existing validation results are preliminary and potentially optimistic.
- **Correction deferred until after midterm submission.** The current model and metrics remain associated with the original manifests. The team plans to verify duplicate relationships, keep connected lesion/duplicate groups together, freeze versioned corrected splits, regenerate features, and retrain/re-evaluate all three models. Updated results will be reported only after those experiments. The held-out cohort remains reserved for final assessment; its overlap/eligibility checks are still required.

- **Probability reliability remains limited.** The CNN has the highest ECE of the three models, although the metadata baseline has a slightly higher Brier score. The validation calibration intercept (−1.82) and slope (1.10), together with the reliability diagram, suggest overestimation of positive-group probabilities. Class weighting is a plausible contributor, but this comparison does not isolate its effect. Post-hoc recalibration is planned; its method and a procedure separating fitting from assessment must be specified. Calculating calibration slope/intercept currently measures calibration and does not apply a correction to predictions.
- **No pretrained backbone.** The CNN was trained from scratch because pretrained weights could not be downloaded in the original environment. Transfer learning is a possible future comparison; a performance improvement has not been established here.


- **No patient identifier in the source data.** HAM10000's metadata gives
  `lesion_id` but no separate patient ID, so the lesion-level train/val
  split cannot
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
- **Diagnosis-specific performance remains planned.** The combined endpoint can conceal errors for individual diagnoses; sensitivity by positive diagnosis, especially melanoma, will be assessed.
- **Not yet evaluated against the held-out test set**, so these numbers
  may not generalise; they are a validation-set estimate only.

## Populations this model should not be used on

Any real patient, in any clinical or non-clinical setting. As a research
prototype it has not been validated on populations outside HAM10000's
source clinics, on skin tones underrepresented in that dataset, on
pediatric patients, or on images taken with equipment/protocols
different from HAM10000's.
