"""Shared label definitions for the HAM10000 binary malignancy task.

Malignant / pre-malignant: melanoma (mel), basal cell carcinoma (bcc),
actinic keratosis / intraepithelial carcinoma (akiec).
Benign: melanocytic nevus (nv), benign keratosis (bkl), dermatofibroma (df),
vascular lesion (vasc).
"""

MALIGNANT_CLASSES = {"mel", "bcc", "akiec"}
BENIGN_CLASSES = {"nv", "bkl", "df", "vasc"}
ALL_CLASSES = MALIGNANT_CLASSES | BENIGN_CLASSES


def dx_to_binary(dx: str) -> int:
    """Map a HAM10000/ISIC diagnosis code to 1 (malignant/pre-malignant) or 0 (benign)."""
    dx = dx.strip().lower()
    if dx in MALIGNANT_CLASSES:
        return 1
    if dx in BENIGN_CLASSES:
        return 0
    raise ValueError(f"Unrecognised diagnosis code: {dx!r}")
