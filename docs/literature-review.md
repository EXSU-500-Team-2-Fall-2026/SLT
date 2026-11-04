# Literature evidence table

Project: Skin Lesion Triage: Comparing Models and Their Calibration (EXSU 500, Team 2)

Created: 2026-11-04

## Purpose and current status

This is a working evidence table for the midterm literature review, not a finished literature review or a list of papers already read by the team. The course requires at least 15 references, predominantly recent primary research, synthesized into an argument. The table itself is a team working aid, not a separate rubric requirement.

The 15 candidate records below were identified from publisher, proceedings, PubMed/PMC, or author-preprint records. The relevance notes are preliminary screening notes from available records/abstracts, not full-text critical appraisals. All team reading statuses start as **Not yet reviewed**. No reader, contribution, numerical result, or full-text appraisal is being claimed on anyone's behalf.

Twelve candidates have publication years 2022–2024; three older papers are retained for foundational dataset, clinical collaboration, and calibration context. This is a starter selection, not a systematic or exhaustive search through November 2026. Replace weaker or redundant candidates after full-text reading, and consider newer 2025–2026 primary studies before finalizing the bibliography. A paper's relevance does not make its results directly comparable to this project's binary endpoint.

## Candidate papers

| ID | Year | Paper and source | Theme / type | Preliminary relevance | Evidence to extract during reading |
|---|---|---|---|---|---|
| P01 | 2018 | [The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions](https://www.nature.com/articles/sdata2018161) | Dataset / foundational | Direct source for HAM10000 provenance and label construction. | Extract image and lesion counts, diagnosis distributions, verification methods, repeated-image structure, and population limits. |
| P02 | 2020 | [Human-computer collaboration for skin cancer recognition](https://pubmed.ncbi.nlm.nih.gov/32572267/) | Clinical motivation / foundational | Studies how AI assistance interacts with clinical expertise and workflow. | Extract study setting, reader design, assistance conditions, and limits on transferring reader-study findings to referral triage. |
| P03 | 2017 | [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html) | Calibration methods / foundational | Methodological basis for assessing probabilities and post-hoc temperature scaling; not a dermoscopy study. | Extract calibration definitions, fitting/evaluation separation, and applicability to a binary weighted-loss CNN. |
| P04 | 2022 | [Validation of artificial intelligence prediction models for skin cancer diagnosis using dermoscopy images: the 2019 International Skin Imaging Collaboration Grand Challenge](https://pmc.ncbi.nlm.nih.gov/articles/PMC9295694/) | External validity / primary research | Evaluates distribution shifts, unfamiliar diagnoses, and image artifacts. | Extract training/test sources, test composition, discrimination changes, and handling of diagnoses outside training categories. |
| P05 | 2022 | [Disparities in Dermatology AI Performance on a Diverse, Curated Clinical Image Set](https://arxiv.org/abs/2203.08807) | Fairness / primary research | DDI study examines performance across diverse skin tones and diseases. Link is the author preprint; obtain the published Science Advances citation before manuscript use. | Extract pathology confirmation, subgroup sizes, external evaluation, and distinction between clinical photographs and dermoscopy. |
| P06 | 2022 | [Uncertainty-aware skin cancer detection: The element of doubt](https://pubmed.ncbi.nlm.nih.gov/35259615/) | Uncertainty / primary research | Studies uncertainty-aware skin cancer models, including MC dropout and Bayesian ensembling. | Extract datasets, split unit, uncertainty definition, calibration/abstention evaluation, and compute requirements. |
| P07 | 2023 | [Learning from dermoscopic images in association with clinical metadata for skin lesion segmentation and classification](https://doi.org/10.1016/j.compbiomed.2022.106321) | Metadata and image models / primary research | Investigates combining dermoscopic features with clinical metadata; journal issue is 2023, DOI contains 2022. | Extract datasets, metadata fields, missingness handling, split unit, and image-only versus multimodal ablations. |
| P08 | 2022 | [A study on skin tumor classification based on dense convolutional networks with fused metadata](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2022.989894/full) | Metadata and CNN models / primary research | Potential justification for comparing metadata and image information. | Extract dataset modalities, baseline comparability, metadata contribution, imbalance handling, and reported uncertainty. |
| P09 | 2023 | [Handling dataset dependence with model ensembles for skin lesion classification from dermoscopic and clinical images](https://doi.org/10.1002/ima.22827) | Generalization / primary research | Examines dataset dependence and ensembles; first online publication was November 2022, journal issue 2023. | Extract within-source versus cross-source evaluation, ensemble cost, and whether clinical and dermoscopic images are analyzed separately. |
| P10 | 2024 | [Few-shot learning for skin lesion classification: A prototypical networks approach](https://www.sciencedirect.com/science/article/pii/S2352914824000765) | Generalization / primary research | Contrasts in-domain and cross-domain learning across dermoscopic domains. | Extract domain definitions, support/test separation, baseline fairness, and domains where gains do not persist. |
| P11 | 2023 | [Comparison of humans versus mobile phone-powered artificial intelligence for the diagnosis and management of pigmented skin cancer in secondary care: a multicentre, prospective, diagnostic, clinical trial](https://doi.org/10.1016/S2589-7500(23)00130-9) | Clinical evaluation / primary research | Prospective comparison relevant to the difference between benchmark classification and clinical management. | Extract recruitment, skin-type inclusion, reference standard, operating points, and diagnosis versus management endpoints. |
| P12 | 2024 | [Accuracy of an artificial intelligence as a medical device as part of a UK-based skin cancer teledermatology service](https://pmc.ncbi.nlm.nih.gov/articles/PMC10996444/) | Triage / primary research | Examines referral decisions and diagnostic sensitivity in a prospective service setting. | Extract cohort, lesion/patient counts, comparator, threshold selection, exclusions, and referral outcomes. |
| P13 | 2024 | [Prospective multicenter study using artificial intelligence to improve dermoscopic melanoma diagnosis in patient care](https://www.nature.com/articles/s43856-024-00598-5) | Clinical evaluation / primary research | Prospective dermoscopy evidence to contrast with retrospective validation. | Extract center composition, melanoma endpoint, AI assistance design, reference standard, and subgroup limitations. |
| P14 | 2024 | [Understanding skin color bias in deep learning-based skin lesion segmentation](https://doi.org/10.1016/j.cmpb.2024.108044) | Segmentation limitations / primary research | Relevant to concerns about region extraction and representation; studies learned segmentation, not this repository's Otsu method. | Extract skin-color measurement methods, datasets, subgroup evaluation, and limits on extending findings to handcrafted segmentation. |
| P15 | 2024 | [DERM12345: A Large, Multisource Dermatoscopic Skin Lesion Dataset with 40 Subclasses](https://pmc.ncbi.nlm.nih.gov/articles/PMC11604664/) | Dataset coverage / primary research | Context for broader diagnostic coverage beyond HAM10000; use the published 40-subclass title rather than the earlier preprint title. | Extract collection setting, diagnostic taxonomy, reference standards, population coverage, and whether it is suitable for any future external test. |

## Team reading and appraisal tracker

A team member should update a row only after actually reading the paper. Suggested sequence: assigned -> reading -> full text read -> appraisal checked -> selected/excluded. Record a reason for exclusion. Use **not reported** for information absent from a paper and **not extracted yet** for information not yet checked.

| ID | Reader | Reading status | Dataset, population, and task | Split unit and leakage safeguards | Models and comparator | Main quantitative result with uncertainty and location | Main limitation | Use in our argument / inclusion decision |
|---|---|---|---|---|---|---|---|---|
| P01 | Unassigned | Not yet reviewed | Not extracted yet | Not extracted yet | Not extracted yet | Not extracted yet | Not appraised yet | Pending |
| P02 | Unassigned | Not yet reviewed | Not extracted yet | Not extracted yet | Not extracted yet | Not extracted yet | Not appraised yet | Pending |
| P03 | Unassigned | Not yet reviewed | Not extracted yet | Not extracted yet | Not extracted yet | Not extracted yet | Not appraised yet | Pending |
| P04 | Unassigned | Not yet reviewed | Not extracted yet | Not extracted yet | Not extracted yet | Not extracted yet | Not appraised yet | Pending |
| P05 | Unassigned | Not yet reviewed | Not extracted yet | Not extracted yet | Not extracted yet | Not extracted yet | Not appraised yet | Pending |
| P06 | Unassigned | Not yet reviewed | Not extracted yet | Not extracted yet | Not extracted yet | Not extracted yet | Not appraised yet | Pending |
| P07 | Unassigned | Not yet reviewed | Not extracted yet | Not extracted yet | Not extracted yet | Not extracted yet | Not appraised yet | Pending |
| P08 | Unassigned | Not yet reviewed | Not extracted yet | Not extracted yet | Not extracted yet | Not extracted yet | Not appraised yet | Pending |
| P09 | Unassigned | Not yet reviewed | Not extracted yet | Not extracted yet | Not extracted yet | Not extracted yet | Not appraised yet | Pending |
| P10 | Unassigned | Not yet reviewed | Not extracted yet | Not extracted yet | Not extracted yet | Not extracted yet | Not appraised yet | Pending |
| P11 | Unassigned | Not yet reviewed | Not extracted yet | Not extracted yet | Not extracted yet | Not extracted yet | Not appraised yet | Pending |
| P12 | Unassigned | Not yet reviewed | Not extracted yet | Not extracted yet | Not extracted yet | Not extracted yet | Not appraised yet | Pending |
| P13 | Unassigned | Not yet reviewed | Not extracted yet | Not extracted yet | Not extracted yet | Not extracted yet | Not appraised yet | Pending |
| P14 | Unassigned | Not yet reviewed | Not extracted yet | Not extracted yet | Not extracted yet | Not extracted yet | Not appraised yet | Pending |
| P15 | Unassigned | Not yet reviewed | Not extracted yet | Not extracted yet | Not extracted yet | Not extracted yet | Not appraised yet | Pending |

## How to complete each appraisal

1. Read the full text and relevant supplements. Record complete authors, publication year, venue, DOI, and publication status in the final bibliography; use the published version when available. Do not count a preprint and its journal version as two references.
2. Extract sample sizes with their units (patients, lesions, or images), class balance, reference standard, modality, and inclusion/exclusion criteria. Distinguish binary malignancy, melanoma-only, multiclass diagnosis, segmentation, and management tasks.
3. Record whether splits are patient-, lesion-, or image-level; whether tuning, calibration, and threshold selection are separate from final evaluation; and whether external cohorts overlap development sources.
4. Extract AUROC, AUPRC, sensitivity/specificity at the stated threshold, and calibration measures when reported. Record confidence intervals or repeated-run variability, and the exact table, figure, or page supporting each result. Do not infer missing metrics or compare headline scores across different cohorts as if they were the same experiment.
5. Write a short appraisal: what the paper supports, what it cannot establish, and the specific consequence for our design. Preserve counter-evidence and negative findings.
6. A second team member should check every numerical claim selected for the manuscript against the source. Only mark a paper team-read when a named member has read it.

## Planned synthesis

- **Clinical motivation and intended use:** P02, P11–P13. Explain the specific dermoscopy workflow and distinguish retrospective classification from improved patient management.
- **Data, representation, and generalization:** P01, P04–P05, P09–P10, P14–P15. Motivate lesion grouping, distribution-shift caution, diagnosis-specific error analysis, and limits on subgroup claims without suitable labels.
- **Model comparison:** P07–P08, supported by P01 and P04. Explain what metadata, handcrafted image features, and learned image features contribute. Find an additional well-matched primary study of classical handcrafted-feature models if these candidates do not adequately support that model choice.
- **Calibration and uncertainty:** P03 and P06. Separate probability calibration, predictive uncertainty, and confidence intervals on performance. Motivate recalibration fitted only on development data and lesion-aware uncertainty estimates.

The final review should connect these themes into the reason for our experiment, rather than give one paragraph per paper. The three-page midterm limit includes references, so prioritize concise synthesis and a compact data/protocol description.

## AI assistance and provenance

Codex assisted with web-based candidate discovery, preliminary relevance notes, and creation of this table on 2026-11-04. Team members must verify bibliography details and perform the recorded reading and critical appraisal themselves. Update the project's Use of Generative AI statement to reflect assistance actually used in the deliverable. No team reading was assumed or fabricated.
