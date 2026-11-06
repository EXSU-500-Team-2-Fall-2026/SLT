# Data-branch pre-merge review

Date: 2026-11-06

Reviewed data snapshot: `efadcbabcc96c43b1a30512c3c00e0504bfbaabf`.
Preserved main snapshot: `9e1f23f647575d0bd8212ab5508489506fe40b96`.

## Scope

This prepares `data/split-audit` for review and eventual merging into `main`. It fixes README fence/indentation formatting and incorporates main's newer documentation, including the known duplicate limitation and decision to defer regrouping, retraining, and reevaluation until after midterm submission.

The model code, split manifests, feature tables, and existing performance results are unchanged. The five crossing near-duplicate pairs remain unresolved and explicitly disclosed.

## Checks completed

- Inspected the new audit/description scripts, known-issues registry, and two test files.
- Ran the assertions in `tests/test_splits.py` and `tests/test_describe_dataset.py` directly against the production modules: **17 cases passed; 1 expected failure reproduced** for the known crossing near-duplicates. Pytest imports/decorators were removed only in an in-memory test loader; fixture arguments and the five parametrized integrity checks were dispatched explicitly. The test files and production code were not changed.
- Ran `python -m src.data.validate_splits --out review-output/split_audit.md`: five manifest integrity checks PASS; five of eighteen known near-duplicate pairs produce WARN; unavailable raw-data checks SKIP. Exit status 0 means no FAIL, not absence of leakage.
- Ran `python -m src.data.describe_dataset --out review-output/dataset`: generated per-class counts from manifests; compared the output with the branch's committed `class_counts.csv`.
- Parsed all five added Python files for syntax.
- Preserved the existing limitation/deferral wording from main in README, model card, and experimental design; no metrics or split counts were edited.
- Fixed the README's stray four-backtick fence and made the audit commands a properly balanced fenced block.

## Verification limits

This was **not a pytest-suite run**. The local bundled Python did not include pytest or scikit-learn; attempts to download them failed because PyPI hostname resolution was unavailable, including on retry with elevated access. The direct assertion run covers the two newly added test files, not the full repository suite or training.

Raw HAM10000 metadata and images were unavailable in this review environment. The contributor-generated full audit remains preserved as `split_audit.md`; image readability, byte hashing, and source-table checks in that saved report were not independently rerun here. The committed demographic tables were inspected but could not be regenerated from raw metadata. Synthetic tests cover relevant counting, missingness, duplicate, and file-check behavior.

The known duplicate expected-failure marker remains intentional until the deferred grouping correction is completed. Nothing in this review establishes that the validation cohort is leakage-free or that existing scores are corrected.

## Remaining before final evaluation

Verify duplicate relationships; implement connected lesion/duplicate groups; version and freeze corrected development splits; regenerate features; retrain and reevaluate all three models; remove the expected-failure marker after the corrected split passes; implement the independent recalibration and lesion-aware uncertainty procedures; complete held-out eligibility and overlap checks. These remain planned after midterm submission.
