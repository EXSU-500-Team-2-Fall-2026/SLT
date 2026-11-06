# Experimental Design

## Research question and clinical motivation

**Primary research question:** How effectively can patient metadata, handcrafted dermoscopic image features, and learned image representations distinguish malignant/premalignant from benign lesions in HAM10000, and how well calibrated are their probability estimates?

The study is motivated by lesion assessment and potential triage support, where failing to recognize potentially serious lesions could delay further assessment. However, this retrospective experiment evaluates classification and probability estimation in an existing dermoscopic dataset. It does not measure referral decisions, clinical benefit, or performance in general population screening.

## Binary classification endpoint

The seven HAM10000 diagnostic categories are grouped as follows:

| Positive: malignant/premalignant (label 1) | Negative: benign (label 0) |
|---|---|
| Melanoma (`mel`) | Melanocytic nevus (`nv`) |
| Basal cell carcinoma (`bcc`) | Benign keratosis (`bkl`) |
| Actinic keratosis / intraepithelial carcinoma (`akiec`) | Dermatofibroma (`df`) |
| | Vascular lesion (`vasc`) |

“Higher-concern” and “lower-concern” may be used as shorthand for these defined groups. They do not represent independently measured clinical concern or referral requirements.

This operational grouping supports a common binary comparison across models. It does not imply identical severity or management within either group. Diagnosis-specific sensitivity, particularly for melanoma, is planned to identify weaknesses that an aggregate positive-class result could conceal.

## Research questions

**RQ1 — Discrimination:** How well does each model distinguish malignant/premalignant from benign lesions?

**RQ2 — Calibration:** How closely do predicted probabilities correspond to observed frequencies of the positive diagnostic group?

**Planned extension to RQ2:** Can post-hoc recalibration improve probability reliability when assessed on cases independent of those used to fit the calibrator?

**Exploratory expectation:** The three approaches may exhibit different discrimination and calibration characteristics. Greater model complexity is not assumed to produce better discrimination or more reliable probabilities.

## Dataset and current splits

HAM10000 contains 10,015 dermoscopic images representing 7,470 lesions. Training and validation are separated by `lesion_id`, using a grouped random split with seed 42. Approximately 15% of lesion groups are allocated to validation; the procedure does not explicitly stratify by diagnosis.

| Partition | Source | Images | Unique lesions | Purpose |
|---|---|---:|---:|---|
| Training | HAM10000 | 8,488 | 6,349 | Model fitting and fitting preprocessing parameters |
| Validation | HAM10000 | 1,527 | 1,121 | Development evaluation and CNN early stopping |
| Held-out test manifest | ISIC 2018 Task 3 distribution obtained through the same Harvard Dataverse record | 1,512 | 1,223 | Planned final evaluation |
| Available held-out images, as documented in the repository | Same held-out distribution | 1,511 | 1,222 | Expected image evaluation cohort, subject to final eligibility checks |

The held-out set is separate from the HAM10000 training/validation division. One test-manifest image, `ISIC_0035068`, is documented as withdrawn from the distributed image collection.

The existing manifest audit found no shared image IDs or lesion IDs across partitions. This does not exclude visually duplicated images under different identifiers or overlap between patients contributing different lesions. Patient-level independence cannot be established from lesion identifiers alone.

Current reported model results are validation results. The reviewed training scripts do not implement the final held-out comparison.

## Implemented models

| Model | Input | Current implementation |
|---|---|---|
| Logistic regression | Age, sex, anatomical location | Median imputation and standardization for age; missing-value handling and one-hot encoding for categorical variables; balanced class weights |
| Histogram gradient boosting | Handcrafted image features | Shape, asymmetry, color, contrast, and texture features; balanced class weights |
| CNN trained from scratch | Dermoscopic image pixels | Four convolutional blocks, global average pooling, and a binary output; 96 × 96 images; training augmentation and a class-weighted loss |

The CNN uses no pretrained backbone. Training augmentation includes random cropping, flips, rotation, and color variation. Validation images undergo resizing and normalization without random augmentation. The CNN checkpoint is selected using validation AUROC.

## Interpretation of model comparisons

- **Model 1 versus Model 2:** metadata-only logistic regression versus gradient boosting using handcrafted image features.
- **Model 2 versus Model 3:** a handcrafted-feature pipeline versus a CNN learning representations from image pixels.
- **All models:** comparison of discrimination and calibration across three approaches.

These comparisons change both the information source and modelling approach. They therefore do not isolate the causal contribution of image information, representation learning, or model complexity.

## Discrimination evaluation

The current evaluation code reports:

- AUROC;
- AUPRC, implemented as average precision;
- sensitivity and specificity at a stated threshold.

The current default classification threshold is **0.5**. This is a reporting convention rather than a clinically validated referral threshold.

Any additional operating-point comparison will use a prespecified criterion, with thresholds selected on development data and frozen before held-out evaluation.

## Calibration evaluation

The current implementation produces:

- reliability diagrams;
- Brier score;
- expected calibration error (ECE), using 10 equal-width probability bins;
- calibration slope and intercept.

These measures will be interpreted together. Brier score measures overall probability prediction error and is not exclusively a measure of calibration. ECE will be treated as supplementary because its value depends on the binning procedure.

**Post-hoc recalibration is planned and is not yet implemented in the reviewed model scripts.** Before implementation, the team will specify the recalibration method and a development-data procedure that separates calibrator fitting from assessment. Improvement measured on the same cases used to fit the calibrator will not be treated as independent evidence.

## Uncertainty and model comparisons

**Current implementation:** 95% percentile bootstrap confidence intervals from 1,000 image-level resamples for AUROC, average precision, sensitivity, specificity, and Brier score.

**Planned improvement:** preserve lesion identifiers with predictions and bootstrap lesions, keeping images from the same lesion together. Model differences will be assessed using paired resampling of the same lesions.

The current confidence intervals do not account for all sources of uncertainty, including repeated CNN training, development-set model selection, or dependence between images of the same lesion. Overlap between separate model confidence intervals will not be used as a formal test of model differences.

## Held-out evaluation checks and final protocol

Before final evaluation, the team will document:

1. **Provenance:** the held-out set’s source, collection context, and relationship to HAM10000.
2. **Overlap:** identifier overlap and, where feasible, exact or near-duplicate images.
3. **Label compatibility:** consistent mapping to the same binary endpoint.
4. **Metadata availability:** whether age, sex, and anatomical location support evaluation of the metadata baseline.
5. **Preprocessing:** consistent application of each trained pipeline, without fitting preprocessing parameters on test data.
6. **Eligibility and exclusions:** handling of the withdrawn image, missing files, and unavailable metadata.
7. **Comparable cohorts:** documentation of which cases each model can evaluate and use of a common eligible cohort for direct paired comparisons.

Models, preprocessing, recalibration procedures, and operating thresholds will be finalized before the held-out comparison. Test outcomes will not be used to revise model selection or thresholds.

## Scope and limitations

This is an educational, retrospective benchmark study. Its conclusions concern the evaluated diagnostic categories, data distributions, and experimental protocol. It does not establish suitability for ordinary smartphone photographs, populations outside the evaluated data, or clinical deployment.

Remaining work includes held-out cohort verification, final evaluation code, an independent recalibration protocol, and lesion-aware uncertainty estimation. These are planned activities rather than completed methods.
