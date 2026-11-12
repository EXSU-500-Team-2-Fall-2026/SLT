# Skin Lesion Triage: Comparing Models and Their Calibration

**EXSU 500: Fundamentals of AI in Medicine, McGill University, Fall 2026**

## Problem

**Research question:** How effectively can patient metadata, handcrafted dermoscopic image features, and learned image representations distinguish malignant/premalignant from benign lesions in HAM10000, and how well calibrated are their probability estimates?

We classify dermoscopic images using an operational binary grouping: **malignant/premalignant** includes melanoma (`mel`), basal cell carcinoma (`bcc`), and actinic keratosis / intraepithelial carcinoma (`akiec`); **benign** includes melanocytic nevus (`nv`), benign keratosis (`bkl`), dermatofibroma (`df`), and vascular lesion (`vasc`). “Higher-concern” and “lower-concern” refer only to these defined groups. They are not measured clinician concern ratings or referral requirements, and do not imply identical severity or management within either group.

This retrospective benchmark is motivated by lesion assessment and potential triage support. It evaluates classification and probability estimation in an existing dermoscopic dataset; it does not measure referral decisions, clinical benefit, general population screening, or suitability for ordinary smartphone photographs.

We compare three implemented models:

1. Logistic regression using age, sex, and anatomical location, with imputation, encoding, and balanced class weights.
2. Histogram gradient boosting using handcrafted image features, with balanced class weights.
3. A four-block convolutional neural network (CNN) with global average pooling, trained from scratch on 96 × 96 images with augmentation and a class-weighted loss. No pretrained backbone is used.

These comparisons change both information source and modelling approach; they do not isolate the causal contribution of image information, representation learning, or model complexity.

- **RQ1 — Discrimination:** how well does each model distinguish the two diagnostic groups? Current measures are AUROC, AUPRC (implemented as average precision), and sensitivity/specificity at threshold 0.5. This threshold is a reporting convention, not a clinically validated referral threshold.
- **RQ2 — Calibration:** how closely do predicted probabilities correspond to observed positive-group frequencies? Current outputs are reliability diagrams, Brier score, ECE using 10 equal-width probability bins, and calibration slope/intercept. Brier score measures overall probability prediction error, not calibration alone; ECE is supplementary and depends on binning.
- **Planned extension:** assess whether post-hoc recalibration improves probability reliability on cases independent of those used to fit the calibrator. Logistic recalibration is used for each model (`src/eval/recalibration.py`: an unpenalised logistic fit on logit-transformed probabilities, clipped at 1e-6). One mapping per model has been fitted on the calibration subset of the duplicate-aware split (`src/eval/fit_calibrators.py`; coefficients in `reports/v2/<model>/calibrator.json`). Whether these mappings improve calibration has not been assessed: that comparison is made once, on held-out cases, and remains future work; see the [agreed protocol](reports/experimental-design.md#future-recalibration-protocol).

The comparison is exploratory: greater model complexity is not assumed to improve either discrimination or calibration.

**Task type:** retrospective supervised binary classification with calibration analysis.

## Dataset

**HAM10000** (Human Against Machine with 10,000 training images): 10,015 dermoscopic images of 7,470 unique pigmented skin lesions, with diagnosis, age, sex and body site.

- **Source:** Harvard Dataverse, [doi:10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T)
- **Licence:** CC BY-NC 4.0 (non-commercial use)
- **Citation:** Tschandl P, Rosendahl C, Kittler H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data* 5, 180161 (2018).

**Splits:** two versions of the HAM10000 development split are committed. Both use `GroupShuffleSplit` with seed 42 and no explicit stratification by diagnosis, and both were fixed without reference to model performance.

*Current, duplicate-aware split* (`data/splits_v2/`, `src/data/make_splits_v2.py`). Images are grouped by connected components of shared `lesion_id` and the 18 known near-duplicate image pairs, which links the 7,470 lesions into 7,452 groups. Whole groups are allocated 85% / 7.5% / 7.5%. None of the 18 pairs crosses subsets. The data lead inspected all 18 pairs visually before the split was frozen (pull request #5).

- **Train:** 8,519 images / 6,350 lesions (19.78% malignant/premalignant). Model fitting.
- **Tune:** 752 images / 560 lesions (18.09%). CNN checkpoint selection and development reporting.
- **Calibration:** 744 images / 560 lesions (17.88%). Fitting the recalibration mappings only; not used for model fitting or selection.

*Original split* (`data/splits/`, `src/data/make_splits.py`). Grouped by `lesion_id` only, with 15% of lesion groups assigned to validation: 8,488 images / 6,349 lesions for training and 1,527 images / 1,121 lesions for validation. Five of the 18 near-duplicate pairs cross its train/validation boundary. It is retained unchanged as the basis of the preliminary results below.

- **Held-out test:** the official ISIC 2018 Task 3 test set, downloaded from the same Dataverse record. Its ground truth lists 1,512 images of 1,223 lesions; 1,511 images (1,222 lesions) are available, because `ISIC_0035068` was withdrawn from this distribution. The manifest audit found zero shared `image_id`/`lesion_id` with the development images. This does not establish absence of visual duplicates or patient overlap. Final evaluation is planned once models, preprocessing, calibrators, and thresholds are frozen; held-out results will not be used to revise those choices. The training and calibrator-fitting scripts refuse the held-out split, and it has not been used.
- **Reference comparison:** reader and CNN data from Tschandl et al., *Nature Medicine* 26, 1229–1234 (2020), included in the same download. Used only for comparison, never for training.
- **Reference comparison:** reader and CNN data from Tschandl et al., *Nature Medicine* 26, 1229–1234 (2020), included in the same download. Used only for comparison, never for training.

Patient-level independence cannot be established from lesion identifiers alone. Before final evaluation, the team will verify held-out provenance and its relationship to HAM10000, duplicate images where feasible, label compatibility, metadata availability for the baseline, preprocessing consistency, exclusions, and a common eligible cohort for paired comparisons.

The data are **not** committed to this repository (see `.gitignore`).

## Known duplicate limitation and its correction

The [data-branch split audit](https://github.com/EXSU-500-Team-2-Fall-2026/SLT/blob/efadcbabcc96c43b1a30512c3c00e0504bfbaabf/reports/dataset/split_audit.md) identified 5 of 18 known near-duplicate image pairs crossing the training/validation boundary of the original split despite having different lesion IDs. Zero shared image IDs or lesion IDs therefore does not establish independence of image content. These pairs provide a potential information-leakage route; the magnitude and direction of their effect on individual metrics have not been measured. Results on the original split are preliminary and potentially optimistic.

**Correction.** The team first decided (2026-11-06) to disclose this limitation and defer the correction. It was then carried out: the duplicate-aware split was added on 2026-11-11 and all three models were retrained on it on 2026-11-12. The grouping rule, seed, and allocation are those stated in the plan; the split was not chosen by model performance. The original manifests, feature tables, and results are preserved and identified as the original split. Hand-crafted features are computed per image and do not depend on the split, so the cached feature tables were reused by `image_id`.

**What the correction does not establish.** The 18 pairs are the published near-duplicates; other visual duplicates may exist. Patient-level independence still cannot be verified. The effect of the five crossing pairs on the original results has not been measured, and results on the two splits are computed on different evaluation images, so their difference is not an estimate of that effect.

The held-out cohort remains reserved for final assessment. The 18 pairs are all HAM10000 development images and do not link development and test. Any newly discovered development/test overlap will be handled using a documented rule before prediction-based evaluation.

See [experimental design](reports/experimental-design.md) for the protocol. Audit code and generated tables have been merged into `main`; the pinned audit link above preserves the original evidence.

## Setup

The code was developed and tested with **Python 3.13**. The pinned package versions in `requirements.txt` do not install on Python 3.9, the version bundled with macOS, so install a current Python first if needed.

```bash
python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                # the tests need no data
```

## Getting the data

**1. Download** the files of the Dataverse record (about 3.1 GB), in either way:

```bash
python -m src.data.download --out data/raw
```

or open [doi:10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T) in a browser, download all files, and note the folder they land in (for example `~/Downloads/dataverse_files`).

**2. Arrange** them into the layout the code expects:

```bash
python -m src.data.prepare_raw                                        # after the download script
python -m src.data.prepare_raw --source ~/Downloads/dataverse_files   # after a browser download
```

This step is needed because the record does not unpack into that layout by itself: the training images come as two archives (`HAM10000_images_part_1.zip` with 5,000 images and `HAM10000_images_part_2.zip` with 5,015) that have no inner folder, two archives contain macOS `__MACOSX` leftovers, and the metadata tables are named differently depending on how they were downloaded. `prepare_raw` works whether the archives are still zipped or already extracted. It never changes the download folder, links already-extracted files instead of copying them (use `--copy` to copy; if you link, keep the download folder where it is), and checks the file counts at the end.

The result is `data/raw/HAM10000_images/` (10,015 `.jpg`), `data/raw/HAM10000_segmentations_lesion_tschandl/` (10,015 masks), `data/raw/ISIC2018_Task3_Test_Images/` (1,511 `.jpg`; the test ground-truth file lists one additional `image_id`, `ISIC_0035068`, that was withdrawn from this Dataverse distribution), `data/raw/HAM10000_metadata.tab` and `data/raw/ISIC2018_Task3_Test_GroundTruth.tab`. `data/splits/*.csv` are already committed, so this step is only needed to retrain a model or rebuild the splits/features from scratch.

## Checking the splits and describing the data

```bash
python -m src.data.validate_splits --hash-images   # audit -> reports/dataset/split_audit.md
python -m src.data.describe_dataset              # counts and demographics -> reports/dataset/*.csv
```

This audit covers the original split (`data/splits/`). It reports the known five cross-split near-duplicate pairs as WARN. A successful exit means no FAIL checks were found; it does not mean the data are free of leakage. Raw-data checks are reported as SKIP when their inputs are unavailable. The duplicate-aware split has its own checks: `make_splits_v2` refuses to write manifests in which an image, lesion, or connected group appears in more than one subset, and `tests/test_splits_v2.py` verifies that no known pair crosses.

The saved [split audit](reports/dataset/split_audit.md) and dataset tables were generated on the data contributor's environment. See [pre-merge review](reports/dataset/premerge-review.md) for independently rerun checks and their limits. The table generator excludes the documented withdrawn test image; demographics are counted per lesion, not per patient.

## Reproducing the results

With the environment from [Setup](#setup) active and the data in `data/raw`.

**Current results (duplicate-aware split, written to `reports/v2/`):**

```bash
python -m src.data.make_splits_v2 --from-splits data/splits --out data/splits_v2   # only to rebuild the committed manifests

python -m src.models.metadata_baseline --splits-dir data/splits_v2 --eval-split tune --predict-splits tune calibration --reports-dir reports/v2/metadata_baseline
python -m src.models.classical_ml      --splits-dir data/splits_v2 --eval-split tune --predict-splits tune calibration --reports-dir reports/v2/classical_ml
python -m src.models.cnn               --splits-dir data/splits_v2 --eval-split tune --predict-splits tune calibration --reports-dir reports/v2/cnn

python -m src.eval.fit_calibrators --reports-dir reports/v2 --models-dir models
```

Each model script trains on the train subset, selects (CNN checkpoint) and reports on the tune subset, and saves its predictions for the tune and calibration subsets with image, lesion, and group IDs. `fit_calibrators` fits one recalibration mapping per model on the calibration predictions only. The scripts refuse the held-out split. `reports/v2/cnn/metrics.json` records every setting of the CNN run (batch size 64, learning rate 1e-3, at most 20 epochs, patience 5, seed 42, device, library version, training time); the reported run stopped after 16 epochs with the checkpoint of epoch 11, in about 11 minutes on an Apple M4 GPU. The CNN uses a GPU when one is available; its results are close but not guaranteed identical across devices and library versions. On the machine used, a second run reproduced all three models exactly.

**Preliminary results (original split, written to `reports/`):**

```bash
# only needed if you don't already have data/features/{train,val}.csv committed:
python -m src.features.build_feature_table --splits train val

python -m src.models.metadata_baseline   # model 1: metadata-only baseline
python -m src.models.classical_ml        # model 2: hand-crafted image features
python -m src.models.cnn                 # model 3: CNN trained from scratch (needs the images in data/raw)
```

With no options, each script trains on `data/splits/train.csv` and reports on `data/splits/val.csv`. `reports/cnn/metrics.json` records that run's image size (96 × 96) and its 15 epochs (early-stopped, about 95 s each) but not its other settings, which were the script's defaults.

Model weights are not committed; each script saves its own under `models/`. Run the tests with `python -m pytest tests/ -q`.

## Scoring a single image

```bash
python -m src.predict path/to/image.jpg --age 62 --sex male --localization back
python -m src.predict --image-id ISIC_0029176   # a HAM10000 image: also prints its true diagnosis
```

`src/predict.py` scores one image with whichever trained models are present in `models/`, using the same preprocessing as the training scripts. For each model it prints the raw probability of the malignant/premalignant group and, if `models/<model>_calibrator.json` exists, the recalibrated probability. Models whose files are missing are skipped. Age, sex and body site are optional and are used only by the metadata baseline. Images from the held-out test set are refused when looked up by ID, because that set is evaluated once, at the end. This is the function an interface would call; it is a research prototype and not for clinical use.

## Results

No held-out test result exists yet. Everything below is development performance.

### Duplicate-aware split, tuning subset

**Development performance on the tuning subset (752 images of 560 lesions)**, before recalibration, threshold 0.5. The CNN checkpoint was selected on this subset, so its values are optimistic to an unmeasured degree. The listed 95% percentile bootstrap CIs use 1,000 image-level resamples. They do not account for dependence between images of the same lesion, repeated training, or development-set model selection. ECE and calibration slope/intercept do not currently have CIs. AUPRC is implemented as average precision.

| Model | AUROC | AUPRC | Sensitivity | Specificity | Brier score | ECE |
|---|---|---|---|---|---|---|
| 1. Metadata baseline (age/sex/site, logistic regression) | 0.803 [0.765, 0.841] | 0.455 [0.377, 0.542] | 0.787 [0.716, 0.854] | 0.698 [0.664, 0.734] | 0.190 [0.176, 0.204] | 0.244 |
| 2. Classical ML (hand-crafted features, gradient boosting) | 0.845 [0.814, 0.874] | 0.510 [0.428, 0.606] | 0.757 [0.683, 0.824] | 0.752 [0.716, 0.783] | 0.160 [0.146, 0.175] | 0.175 |
| 3. CNN (trained from scratch, no pretrained backbone) | 0.872 [0.848, 0.897] | 0.537 [0.458, 0.628] | 0.926 [0.882, 0.969] | 0.680 [0.645, 0.718] | 0.172 [0.157, 0.186] | 0.216 |

The CNN has the highest observed AUROC, AUPRC, and sensitivity; histogram gradient boosting has the highest specificity and the lowest Brier score and ECE. These are descriptive differences; separate confidence-interval overlap does not establish statistical superiority, equivalence, or absence of a difference. Paired lesion-group bootstrap comparisons are planned. Input information and model family change together, so the observed differences cannot be attributed to image information alone.

The calibration intercepts are negative for all three models (−1.56, −1.34, −1.78 for models 1–3; slopes 1.02, 0.91, 1.19), and the reliability diagrams (`reports/v2/<model>/reliability_diagram.png`) show predicted probabilities above observed frequencies. This suggests systematic overestimation of positive-group probabilities. Class weighting is a plausible contributor, but its causal effect has not been isolated by an ablation.

**Fitted recalibration mappings** (calibration subset, 744 images, 133 positive; intercept and slope on the logit scale): −1.458 and 0.694 for model 1, −1.290 and 0.865 for model 2, −1.408 and 0.976 for model 3. These coefficients describe the fitted correction. They are not evidence that recalibration improves calibration, which has not been evaluated.

Full outputs are in `reports/v2/<model>/`: `metrics.json`, the tune and calibration predictions, `calibrator.json`, and the reliability diagram.

### Original split, validation subset (preliminary)

**Preliminary validation-set performance on the original split (1,527 images), which has the known cross-split near-duplicate limitation**, threshold 0.5, with the same kind of image-level bootstrap CIs. These results are kept for the record and are superseded by the table above. The two tables use different evaluation images and are not directly comparable.

| Model | AUROC | AUPRC | Sensitivity | Specificity | Brier score | ECE |
|---|---|---|---|---|---|---|
| 1. Metadata baseline (age/sex/site, logistic regression) | 0.787 [0.761, 0.813] | 0.424 [0.376, 0.481] | 0.800 [0.755, 0.845] | 0.688 [0.663, 0.712] | 0.199 | 0.241 |
| 2. Classical ML (hand-crafted features, gradient boosting) | 0.840 [0.818, 0.862] | 0.555 [0.502, 0.616] | 0.747 [0.697, 0.799] | 0.730 [0.708, 0.756] | 0.160 | 0.163 |
| 3. CNN (trained from scratch, no pretrained backbone) | 0.850 [0.830, 0.869] | 0.528 [0.472, 0.588] | 0.920 [0.889, 0.949] | 0.643 [0.617, 0.668] | 0.197 | 0.253 |

Calibration intercepts on this subset were −1.47, −1.22, and −1.82 for models 1–3, with slopes near 1.

The CNN was trained from scratch because the original training environment could not download pretrained weights. Transfer learning is a possible future comparison; improved performance has not been demonstrated here.

## Planned evaluation work

Completed since the first version of this plan: duplicate-aware grouping, versioned development splits, retraining of all three models, predictions saved with image, lesion, and group IDs, and one fitted recalibration mapping per model. Still to do:

- Implement lesion-group bootstrap CIs and paired model-difference estimates.
- Complete the [agreed recalibration protocol](reports/experimental-design.md#future-recalibration-protocol): original-versus-recalibrated held-out comparisons. Primary comparison: change in Brier score, supported by reliability diagrams, slope/intercept, ECE, discrimination metrics, and paired lesion-group bootstrap CIs. Do not claim improvement from fitting cases alone or revise methods based on test scores.
- Report diagnosis-specific sensitivity, particularly for melanoma.
- Verify the held-out cohort and implement final evaluation for all models, including metadata compatibility and explicit missing-input/exclusion handling.
- Prespecify any additional operating-point criterion using development data, and freeze thresholds before final testing.

These are planned activities, not completed methods or results.

## Team roles

| Role | Member |
|---|---|
| Data lead | Giuliana Curcio |
| Modelling lead | Abdul Aziz Mourad |
| Interface lead | Sophia Lee |
| Writing lead | Ariel Subekti |

See [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md) for the CRediT contribution statement, and [`MODEL_CARD.md`](MODEL_CARD.md) for the current best model's intended use, data, metrics, and limitations.

## Use of Generative AI

*Draft: each team member checks the uses listed for their own work and fills in the versions before submission.*

The team used the following generative AI tools:

- **Claude** (Anthropic, claude.ai, [version]): comparing candidate datasets, drafting and revising this README, planning the evaluation protocol, drafting and editing the midterm report, identifying candidate references, and reviewing code and data checks.
- **Claude Code** (Anthropic, [version]): writing and debugging the data download, split, feature, model, evaluation and test code, and drafting `MODEL_CARD.md` and `CONTRIBUTIONS.md`.
- **ChatGPT** (OpenAI, [version]): drafting an early project plan.
- **OpenAI Codex** ([version]): finding candidate papers, building the literature evidence table in `reports/literature-review.md`, helping revise the experimental design, and aligning repository documentation with implemented and planned methods.
- **Google Gemini** ([version]): explaining one candidate paper during the literature review.

All AI-generated code was reviewed and tested by the authors, every reference was checked against the original publication by at least one author, and no data or results were generated or altered by an AI tool. The authors take full responsibility for this work.

## Licence

Code: MIT (see `LICENSE`). Data: CC BY-NC 4.0, owned by the dataset authors and not redistributed here.

*Course prototype for research and educational purposes. Not a medical device and not for clinical use.*
