# Experimental Design

## Research question and clinical motivation
**Clinical problem:** Distinguishing malignant/premalignant from benign skin lesions is challenging because they can share similar visual features. AI may support lesion assessment, but accurate classification does not always mean reliable, predicted probabilities.

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

## Known duplicate limitation and midterm decision

The [data-branch split audit](https://github.com/EXSU-500-Team-2-Fall-2026/SLT/blob/efadcbabcc96c43b1a30512c3c00e0504bfbaabf/reports/dataset/split_audit.md) identifies 5 of 18 known near-duplicate image pairs crossing the current training/validation boundary despite having different lesion IDs. Zero shared image IDs or lesion IDs therefore does not establish independence of image content. These pairs provide a potential information-leakage route; the magnitude and direction of their effect on individual metrics have not been measured. Existing validation results are preliminary and potentially optimistic.

**Midterm decision (2026-11-06):** retain the current split manifests, trained-model results, and feature tables for the midterm, disclose this limitation, and defer correction of the grouping rule and rerunning experiments until after the midterm submission. No corrected split or corrected performance is claimed.

This finding qualifies the identifier-disjointness statement above: distinct identifiers do not guarantee distinct lesion content. The original split and results will remain traceable rather than being described as corrected.

### Planned correction after midterm submission

1. Verify the published near-duplicate relationships and visually inspect flagged pairs; record uncertain matches.
2. Create connected groups linking images that share a lesion ID or a confirmed near-duplicate relationship. Transitive links must remain together.
3. Rebuild development partitions using these combined groups and seed 42, starting with approximately 85% training, 7.5% tuning/validation, and 7.5% calibration groups, as specified in the future recalibration protocol below. Audit class/group counts before freezing. The grouping change may alter image/lesion counts and class proportions. Do not select a split based on model performance.
4. Require no shared image IDs, lesion IDs, or confirmed duplicate groups across partitions. Recheck class balance, missingness, and image eligibility, then freeze and version corrected manifests while preserving the originals.
5. Regenerate feature tables and metadata joins; refit training-derived preprocessing and retrain all three models, including validation-based CNN checkpoint selection.
6. Regenerate metrics, uncertainty estimates, and calibration artifacts, explicitly identifying the split version and training settings. Do not merely remove validation images and relabel the old results as corrected.

The five reported crossing pairs concern training/validation, so they do not by themselves require changing the held-out cohort. Continue the held-out overlap audit. If development/test overlap is found, prespecify and document its handling before evaluating predictions, preserving test independence. Recalibration fitting/assessment and final held-out comparison will follow the corrected development workflow.

The audit implementation and generated tables have been integrated into `main`. The pinned report above preserves the original audit evidence.

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

**Logistic recalibration has been selected as the future method for all three models and is not yet implemented in the reviewed scripts.** A separate intercept and slope will be fitted to each model's logit-transformed probabilities on disjoint calibration data without class balancing. Independent before/after assessment will use the frozen held-out cohort. Improvement measured on calibrator-fitting cases alone will not be treated as independent evidence.

### Future recalibration protocol

**Status:** planned, not implemented. Execute after midterm submission and duplicate-aware regrouping; existing manifests, feature tables, and preliminary results remain unchanged. Action owners will be discussed with the team and are not assigned here.

#### 1. Freeze disjoint development subsets

Use connected groups joining shared lesion IDs and confirmed near-duplicate relationships. The starting allocation is:

| Future subset | Approximate share of corrected HAM10000 development groups | Permitted use |
|---|---:|---|
| Training | 85% | Fit preprocessing and underlying models |
| Tuning/validation | 7.5% | Model/hyperparameter selection and CNN checkpoint selection |
| Calibration | 7.5% | Fit a separate probability mapping for each frozen model |

These are proportions of combined groups, not promised image counts, and exclude the separate held-out evaluation cohort. Use seed 42, audit group disjointness, and inspect positive/negative group counts and metadata availability before freezing the allocation. If counts make a subset infeasible, document a revised allocation before model fitting; do not search for splits yielding favorable scores. This supersedes the earlier planned 85/15 development split for future experiments only. The current 8,488/1,527 image split remains the midterm split.

Save versioned manifests and a split audit with group, lesion, image, and class counts. Calibration cases must not influence model fitting, preprocessing fitting, early stopping, hyperparameter selection, or feature selection. Fit training-derived preprocessing on training data only; use deterministic evaluation preprocessing for tuning, calibration, and test cases.

#### 2. Retrain, select, and freeze the underlying models

Regenerate handcrafted features and metadata joins for the corrected partitions. Retrain all three model families. Use tuning data for development choices and CNN checkpoint selection, then freeze each fitted model and its preprocessing. Preserve image IDs, lesion IDs, combined-group IDs, labels, original probabilities, and split/model versions with prediction outputs.

Do not refit the underlying models after calibrator fitting. Any later model refit invalidates the associated calibrator and requires a new development-only calibration step before test evaluation.

#### 3. Fit logistic recalibration on calibration data only

For each frozen model, fit a separate logistic mapping:

`p_cal = sigmoid(a + b * logit(p_original))`

The intercept adjusts the overall probability level; the slope adjusts the spread of the log-odds. Clip probabilities away from 0 and 1 before the logit and record the numerical clipping rule, solver, regularization choice, and fitted coefficients in the reproducibility record. This is logistic recalibration of probabilities, not merely the existing diagnostic slope/intercept calculation and not a new image classifier.

Fit using the observed calibration outcomes without balanced class weights, over/undersampling, or artificial prevalence balancing. The underlying model's training weights remain a separate design choice. Calibration fitting uses the image-level target consistent with reported image-level metrics; preserve grouping for uncertainty analysis.

Do not claim calibration improvement using only these fitting cases. Check numerical convergence and record failures or insufficient class support using a prespecified development-only handling rule before accessing test performance. The method is common across all three models; do not select calibrator families using held-out scores.

#### 4. Freeze the evaluation plan before held-out predictions

Complete provenance, duplicate-overlap, label-compatibility, metadata, preprocessing, and exclusion checks. Establish and record a common eligible held-out cohort for direct model comparisons. Prespecify the numerical fitting settings, failure handling, binning, bootstrap settings, and any threshold policy using development information only.

Freeze underlying models, calibrators, preprocessing, model-selection decisions, thresholds, and the evaluation cohort before the final comparison. The held-out cohort remains reserved; its outcomes must not fit calibrators, choose methods, or revise model/threshold selection.

#### 5. Compare original and recalibrated probabilities on the same held-out cases

| Role | Prespecified comparison |
|---|---|
| Primary before/after outcome | Brier-score difference: recalibrated minus original; negative favors recalibration |
| Supporting calibration evidence | Original/recalibrated reliability diagrams and calibration slope/intercept |
| Supplementary measure | ECE using the same 10 equal-width probability bins |
| Continued discrimination reporting | AUROC and average precision before and after recalibration |
| Operating-point reporting | Sensitivity/specificity under a documented threshold rule |
| Uncertainty | Paired lesion-group bootstrap intervals for before/after metric differences |

Brier score assesses overall probability error, not calibration alone. Interpret its difference alongside the calibration artifacts. Estimate performance at the image level; resample lesion groups together, retaining all their images. Where confirmed duplicate links connect different lesion IDs, keep the full connected group together in resampling too. Use the same sampled groups for original/recalibrated predictions and for paired comparisons across models on their common cohort. Document that intervals condition on the frozen fitted models and calibrators and do not capture all training variability.

The existing 0.5 threshold is a reporting convention. Applying 0.5 to original and recalibrated probabilities may change sensitivity and specificity; describe that as a change in operating point, not automatically as improved triage. Any additional operating-point policy must be fixed before held-out evaluation.

Report improvement, deterioration, or no clear evidence of change. Do not use test results to refit the models/calibrators, choose a different recalibration method, or change thresholds. Preserve the original preliminary midterm outputs and report the corrected experiments separately.

#### Future action checklist

- [ ] Confirm duplicate relationships and create connected groups.
- [ ] Inspect subset feasibility; freeze and version training/tuning/calibration manifests and audits.
- [ ] Regenerate inputs and retrain all three models using only the permitted subsets.
- [ ] Freeze models/preprocessing and save aligned calibration predictions.
- [ ] Finalize numerical recalibration settings and failure handling; implement and test the logistic mappings.
- [ ] Fit and save each calibrator on calibration data without class balancing.
- [ ] Complete held-out cohort checks and freeze the before/after evaluation plan.
- [ ] Evaluate original and recalibrated predictions on identical held-out cases; generate paired uncertainty estimates and calibration artifacts.
- [ ] Report all outcomes and update documentation without overwriting preliminary results.

Responsibilities and team deadlines remain to be agreed. No owner is inferred from the existing role list.

Method references: [probability calibration guidance](https://scikit-learn.org/stable/modules/calibration.html); [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html) for calibration context and the distinction from single-parameter temperature scaling.

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

The known cross-split near-duplicates limit the interpretation of existing validation results, independently of the image-level bootstrap limitation. Correction and retraining are deferred until after the midterm submission.

Remaining work includes duplicate-aware regrouping, corrected split versioning and retraining of all three models, held-out cohort verification, final evaluation code, an independent recalibration protocol, and lesion-aware uncertainty estimation. These are planned activities rather than completed methods.
