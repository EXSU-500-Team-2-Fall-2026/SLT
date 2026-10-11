# Contributions

## CRediT author statement

Roles follow the [CRediT taxonomy](https://credit.niso.org/). This is a
living draft, updated as work progresses; it will be finalised against the
actual commit history at submission (see §5.4 of the course specification:
commit history is part of the assessment).

| Contributor | Team role | CRediT roles so far |
|---|---|---|
| Giuliana Curcio | Data lead | Data curation; Software (dataset download script, lesion-level split builder) |
| Abdul Aziz Mourad | Modelling lead | Methodology; Software (baseline and classical-ML models, shared evaluation harness); Formal analysis; Validation |
| Sophie Lee | Interface lead | *Pending — interface not yet built* |
| Ariel Subekti | Writing lead | *Pending — manuscript not yet started* |

Every team member is expected to also contribute to the manuscript and to
review the full codebase before submission, per the course specification
("every member is expected to contribute to the code and to the
manuscript; the roles establish who is answerable for each area, not who
does all of it").

## Use of Generative AI

Claude Code (Anthropic) was used throughout development as a coding
assistant, under direct human direction from the modelling lead:

- Writing and debugging Python code (data download, feature extraction,
  model training scripts, evaluation metrics, tests) from requirements
  specified in conversation.
- Running and interpreting training/evaluation output, including flagging
  issues such as the blocked default `urllib` User-Agent on the Harvard
  Dataverse API, the mismatch between `make_splits.py` and the actual
  `.tab` file format, and the absence of uncertainty quantification
  against the course specification.
- Drafting this contribution statement and the data/reproduction sections
  of `README.md`.

All methodological choices (model families, features, splits, metrics,
thresholds) were directed and reviewed by the modelling lead. No
generative AI was used to fabricate, alter, or interpret results, and no
dataset content or identifiable information was shared with any external
service beyond what the dataset's own licence permits. As stated in the
course specification, the team remains fully responsible for everything
submitted, including any AI-assisted code or text.
