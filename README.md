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
- **Planned extension:** assess whether post-hoc recalibration improves probability reliability on cases independent of those used to fit the calibrator. The method and independent assessment procedure are not yet implemented or finalized.

The comparison is exploratory: greater model complexity is not assumed to improve either discrimination or calibration.

**Task type:** retrospective supervised binary classification with calibration analysis.

## Dataset

**HAM10000** (Human Against Machine with 10,000 training images): 10,015 dermoscopic images of 7,470 unique pigmented skin lesions, with diagnosis, age, sex and body site.

- **Source:** Harvard Dataverse, [doi:10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T)
- **Licence:** CC BY-NC 4.0 (non-commercial use)
- **Citation:** Tschandl P, Rosendahl C, Kittler H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data* 5, 180161 (2018).

**Splits:** the repository records these as declared before modelling. HAM10000 train/validation partitions use `GroupShuffleSplit` grouped by `lesion_id`, with seed 42 and 15% of lesion groups assigned to validation; diagnosis is not explicitly stratified. The separate held-out distribution is not part of an 85/15/15 division. See `src/data/make_splits.py` and committed manifests in `data/splits/`.

- **Train:** 8,488 images / 6,349 lesions.
- **Validation:** 1,527 images / 1,121 lesions. Used for current development evaluation and CNN early stopping. Current reports use threshold 0.5; a protocol separating recalibration fitting from independent assessment remains to be specified using development data.
- **Held-out test:** the official ISIC 2018 Task 3 test set, downloaded from the same Dataverse record. Its ground truth lists 1,512 images of 1,223 lesions; 1,511 images (1,222 lesions) are available, because `ISIC_0035068` was withdrawn from this distribution. The manifest audit found zero shared `image_id`/`lesion_id` with train or validation. This does not establish absence of visual duplicates or patient overlap. Final evaluation is planned once models, preprocessing, calibrators, and thresholds are frozen; held-out results will not be used to revise those choices. The reviewed training scripts currently evaluate validation only.
- **Reference comparison:** reader and CNN data from Tschandl et al., *Nature Medicine* 26, 1229–1234 (2020), included in the same download. Used only for comparison, never for training.

Patient-level independence cannot be established from lesion identifiers alone. Before final evaluation, the team will verify held-out provenance and its relationship to HAM10000, duplicate images where feasible, label compatibility, metadata availability for the baseline, preprocessing consistency, exclusions, and a common eligible cohort for paired comparisons.

The data are **not** committed to this repository (see `.gitignore`).

## Known duplicate limitation and deferred correction

The [data-branch split audit](https://github.com/EXSU-500-Team-2-Fall-2026/SLT/blob/efadcbabcc96c43b1a30512c3c00e0504bfbaabf/reports/dataset/split_audit.md) identifies 5 of 18 known near-duplicate image pairs crossing the current training/validation boundary despite having different lesion IDs. Zero shared image IDs or lesion IDs therefore does not establish independence of image content. These pairs provide a potential information-leakage route; the magnitude and direction of their effect on individual metrics have not been measured. Existing validation results are preliminary and potentially optimistic.

**Midterm decision (2026-11-06):** retain the current split manifests, trained-model results, and feature tables for the midterm, disclose this limitation, and defer correction of the grouping rule and rerunning experiments until after the midterm submission. No corrected split or corrected performance is claimed.

After submission, the team plans to verify the published duplicate relationships and group images using connected components of shared lesion IDs and confirmed near-duplicate links. It will retain seed 42 and approximately 15% of combined groups for validation, audit and freeze the corrected manifests, regenerate development features, and retrain/re-evaluate all three models. The original manifests and results will be preserved and identified by version. The corrected split will not be chosen by model performance.

The held-out cohort remains reserved for final assessment. These five pairs concern train/validation and do not by themselves require changing the held-out cohort. Any newly discovered development/test overlap will be handled using a documented rule before prediction-based evaluation. Recalibration and final testing will follow the corrected development workflow.

See [experimental design](reports/experimental-design.md) for current methods and planned work. Audit code and generated tables are currently on `data/split-audit`; the pinned audit link above records the evidence without implying those additions have been merged.

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
python -m src.models.cnn                 # model 3: CNN trained from scratch (needs the images in data/raw)

python -m pytest tests/ -q
```

`reports/cnn/metrics.json` records the reported CNN run's image size (96 × 96) and its 15 epochs (early-stopped, about 95 s each). It does not record the other settings, so the command above assumes the script's defaults: batch size 64, learning rate 1e-3, at most 20 epochs, patience 5, seed 42. Model weights are not committed; each script saves its own under `models/`.

Each model script trains on `data/splits/train.csv`, reports on `data/splits/val.csv` only (the test set is touched once, at the end, across all models), and writes its metrics/reliability diagram to `reports/<model>/`.

## Results

**Preliminary validation-set performance on the original split with the known cross-split near-duplicate limitation**, threshold 0.5. No performance correction or retraining has been carried out for this finding. The listed 95% percentile bootstrap CIs use 1,000 image-level resamples. They do not account for dependence between images of the same lesion, repeated training, or development-set model selection. ECE and calibration slope/intercept do not currently have CIs. AUPRC is implemented as average precision.

| Model | AUROC | AUPRC | Sensitivity | Specificity | Brier score | ECE |
|---|---|---|---|---|---|---|
| 1. Metadata baseline (age/sex/site, logistic regression) | 0.787 [0.761, 0.813] | 0.424 [0.376, 0.481] | 0.800 [0.755, 0.845] | 0.688 [0.663, 0.712] | 0.199 | 0.241 |
| 2. Classical ML (hand-crafted features, gradient boosting) | 0.840 [0.818, 0.862] | 0.555 [0.502, 0.616] | 0.747 [0.697, 0.799] | 0.730 [0.708, 0.756] | 0.160 | 0.163 |
| 3. CNN (trained from scratch, no pretrained backbone) | 0.850 [0.830, 0.869] | 0.528 [0.472, 0.588] | 0.920 [0.889, 0.949] | 0.643 [0.617, 0.668] | 0.197 | 0.253 |

The CNN has the highest observed validation AUROC, while histogram gradient boosting has higher observed AUPRC and lower Brier score and ECE. These are descriptive differences; separate confidence-interval overlap does not establish statistical superiority, equivalence, or absence of a difference. Paired lesion-level bootstrap comparisons are planned. Input information and model family change together, so the observed differences cannot be attributed to image information alone.

The negative calibration intercepts (−1.47, −1.22, −1.82 for models 1–3), slopes near 1, and reliability diagrams suggest systematic overestimation of positive-group probabilities on validation. Class weighting is a plausible contributor, but its causal effect has not been isolated by an ablation. The CNN has the highest ECE, while the metadata baseline has a slightly higher Brier score than the CNN; calibration and overall probability error should therefore be interpreted using multiple measures.

The CNN was trained from scratch because the original training environment could not download pretrained weights. Transfer learning is a possible future comparison; improved performance has not been demonstrated here.

## Planned evaluation work

- After midterm submission, implement duplicate-aware grouping, version and freeze corrected development splits, regenerate features, and retrain/re-evaluate all three models; preserve original results as preliminary.
- Preserve image and lesion IDs with predictions; implement lesion-level bootstrap CIs and paired model-difference estimates.
- Specify and implement recalibration with independent assessment; do not evaluate improvement solely on calibrator-fitting cases.
- Report diagnosis-specific sensitivity, particularly for melanoma.
- Verify the held-out cohort and implement final evaluation for all models, including metadata compatibility and explicit missing-input/exclusion handling.
- Prespecify any additional operating-point criterion using development data, and freeze thresholds before final testing.

These are planned activities, not completed methods or results.

## Team roles

| Role | Member |
|---|---|
| Data lead | Giuliana Curcio |
| Modelling lead | Abdul Aziz Mourad |
| Interface lead | Sophie Lee |
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
