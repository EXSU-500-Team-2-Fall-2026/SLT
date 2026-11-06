"""Known data issues in HAM10000 and the ISIC 2018 Task 3 test set that the
split audit (src/data/validate_splits.py) and the dataset description
(src/data/describe_dataset.py) need to know about.

NEAR_DUPLICATE_PAIRS
    18 pairs of near-identical HAM10000 images that carry *different*
    lesion_id values, so a lesion-level split cannot keep them together on
    its own. Published by Abhishek, Jain and Hamarneh, "Investigating the
    quality of DermaMNIST and Fitzpatrick17k dermatological image datasets",
    Scientific Data 12, 196 (2025), doi:10.1038/s41597-025-04382-5. List
    taken from the authors' repository:
    https://github.com/kakumarabhishek/Corrected-Skin-Image-Datasets/blob/main/DermaMNIST/HAM10000_DuplicateConfirmation/AnalysesOutputFiles/newly_discovered_duplicates.csv

WITHDRAWN_TEST_IMAGES
    Image IDs listed in the ISIC 2018 Task 3 test ground truth that are not
    in the Harvard Dataverse distribution of the test images.
"""

NEAR_DUPLICATE_PAIRS: list[tuple[str, str]] = [
    ("ISIC_0033481", "ISIC_0033421"),
    ("ISIC_0033556", "ISIC_0032634"),
    ("ISIC_0029152", "ISIC_0033173"),
    ("ISIC_0029036", "ISIC_0027027"),
    ("ISIC_0027820", "ISIC_0028209"),
    ("ISIC_0027689", "ISIC_0032647"),
    ("ISIC_0030906", "ISIC_0024879"),
    ("ISIC_0033391", "ISIC_0034186"),
    ("ISIC_0025226", "ISIC_0030074"),
    ("ISIC_0026087", "ISIC_0025664"),
    ("ISIC_0024736", "ISIC_0029748"),
    ("ISIC_0024770", "ISIC_0027811"),
    ("ISIC_0027162", "ISIC_0029061"),
    ("ISIC_0024602", "ISIC_0032283"),
    ("ISIC_0029625", "ISIC_0024629"),
    ("ISIC_0030000", "ISIC_0031392"),
    ("ISIC_0024437", "ISIC_0031299"),
    ("ISIC_0033374", "ISIC_0033417"),
]

WITHDRAWN_TEST_IMAGES: frozenset[str] = frozenset({"ISIC_0035068"})
