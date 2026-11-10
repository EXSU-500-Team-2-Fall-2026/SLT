"""Arrange a HAM10000 download into the folder layout the rest of the code expects.

The Harvard Dataverse record (doi:10.7910/DVN/DBW86T) does not unpack into
that layout by itself:

- the training images come as two archives, HAM10000_images_part_1.zip
  (5,000 images) and HAM10000_images_part_2.zip (5,015 images), with the
  .jpg files at the top level of each archive;
- the test-image and segmentation archives contain macOS "__MACOSX"
  folders and "._*" files that are not images;
- the metadata and test ground-truth tables are named differently
  depending on how they were downloaded (HAM10000_metadata,
  HAM10000_metadata.tab, ISIC2018_Task3_Test_GroundTruth.csv, ...).

This script reads a download folder, whether the archives are still zipped
or already extracted, and produces:

    <out>/HAM10000_images/                           10,015 .jpg
    <out>/HAM10000_segmentations_lesion_tschandl/    10,015 .png
    <out>/ISIC2018_Task3_Test_Images/                 1,511 .jpg
    <out>/HAM10000_metadata.tab
    <out>/ISIC2018_Task3_Test_GroundTruth.tab

The two tables keep their original contents; only the file name changes.
They may be comma- or tab-separated, and every loader in this repository
detects the separator.

Files already extracted are linked (symbolic links) by default so that
3 GB of images are not duplicated; use --copy to copy them. If you link,
do not move or delete the download folder afterwards. Files inside .zip
archives are extracted. Nothing in the download folder is changed or
deleted, existing files in <out> are left as they are, and the script can
be run again safely.

Usage:
    # after `python -m src.data.download --out data/raw`
    python -m src.data.prepare_raw

    # after a manual "Download all" from the Dataverse page
    python -m src.data.prepare_raw --source ~/Downloads/dataverse_files
    python -m src.data.prepare_raw --source ~/Downloads/dataverse_files --copy
"""
import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

IMAGE_SETS = [
    # (name prefix in the download, output folder, file suffix, expected count, required)
    ("HAM10000_images", "HAM10000_images", ".jpg", 10015, True),
    ("ISIC2018_Task3_Test_Images", "ISIC2018_Task3_Test_Images", ".jpg", 1511, True),
    ("HAM10000_segmentations_lesion_tschandl", "HAM10000_segmentations_lesion_tschandl", ".png", 10015, False),
]
TABLES = [
    # (base name in the download, output file name, required)
    ("HAM10000_metadata", "HAM10000_metadata.tab", True),
    ("ISIC2018_Task3_Test_GroundTruth", "ISIC2018_Task3_Test_GroundTruth.tab", True),
    (
        "ISIC2018_Task3_Test_NatureMedicine_AI_Interaction_Benefit",
        "ISIC2018_Task3_Test_NatureMedicine_AI_Interaction_Benefit.tab",
        False,
    ),
]
TABLE_EXTENSIONS = (".tab", ".csv", ".tsv", "")


def is_junk(path_in_archive_or_folder: str) -> bool:
    """macOS archive leftovers: __MACOSX folders, ._ resource forks, .DS_Store."""
    parts = Path(path_in_archive_or_folder).parts
    return any(p == "__MACOSX" or p.startswith("._") or p == ".DS_Store" for p in parts)


def _place(src: Path, dest: Path, copy: bool) -> None:
    if copy:
        shutil.copy2(src, dest)
        return
    try:
        os.symlink(src.resolve(), dest)
    except OSError:  # e.g. Windows without permission to create links
        shutil.copy2(src, dest)


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def gather_files(source: Path, prefix: str, dest_dir: Path, suffix: str, copy: bool) -> int:
    """Collect every `suffix` file from folders and .zip archives in `source`
    whose name starts with `prefix`. Returns the number of files added."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    added = 0
    entries = sorted(p for p in source.iterdir() if p.name.startswith(prefix))

    for folder in (p for p in entries if p.is_dir()):
        if folder.resolve() == dest_dir.resolve():
            continue
        for src in sorted(folder.rglob(f"*{suffix}")):
            if is_junk(str(src.relative_to(folder))) or not src.is_file():
                continue
            dest = dest_dir / src.name
            if not _exists(dest):
                _place(src, dest, copy)
                added += 1

    for archive in (p for p in entries if p.is_file() and p.suffix.lower() == ".zip"):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                if member.is_dir() or is_junk(member.filename) or not member.filename.lower().endswith(suffix):
                    continue
                dest = dest_dir / Path(member.filename).name
                if _exists(dest):
                    continue
                with zf.open(member) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
                added += 1
    return added


def gather_loose_training_images(source: Path, dest_dir: Path, copy: bool) -> int:
    """Training images unzipped straight into `source` (the two training
    archives have no inner folder, so a plain `unzip` leaves them loose)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    added = 0
    for src in sorted(source.glob("ISIC_*.jpg")):
        dest = dest_dir / src.name
        if not _exists(dest):
            _place(src, dest, copy)
            added += 1
    return added


def place_table(source: Path, base: str, dest: Path, copy: bool) -> str:
    """Returns 'present', 'added' or 'missing'."""
    if _exists(dest):
        return "present"
    for ext in TABLE_EXTENSIONS:
        candidate = source / f"{base}{ext}"
        if candidate.is_file() and candidate.resolve() != dest.resolve():
            _place(candidate, dest, copy)
            return "added"
    return "missing"


def count_files(folder: Path, suffix: str) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for name in os.listdir(folder) if name.lower().endswith(suffix) and not is_junk(name))


def prepare(source: Path, out: Path, copy: bool = False, expected: dict | None = None) -> list[str]:
    """Build the layout in `out`. Returns a list of problems (empty if complete).
    `expected` maps output folder name -> expected file count (defaults to the
    counts of the real dataset)."""
    source, out = Path(source).expanduser(), Path(out).expanduser()
    if not source.is_dir():
        return [f"download folder not found: {source}"]
    out.mkdir(parents=True, exist_ok=True)
    problems = []

    for prefix, folder, suffix, default_count, required in IMAGE_SETS:
        dest_dir = out / folder
        added = gather_files(source, prefix, dest_dir, suffix, copy)
        if folder == "HAM10000_images":
            added += gather_loose_training_images(source, dest_dir, copy)
        found = count_files(dest_dir, suffix)
        want = default_count if expected is None else expected.get(folder, default_count)
        status = "ok" if found == want else ("INCOMPLETE" if required else "incomplete (optional)")
        print(f"{folder + '/':<45} {found:>6} of {want:<6} {suffix} files  (+{added} new)  {status}")
        if found != want and required:
            problems.append(f"{folder}: found {found} {suffix} files, expected {want}")

    for base, name, required in TABLES:
        result = place_table(source, base, out / name, copy)
        print(f"{name:<45} {result}")
        if result == "missing" and required:
            problems.append(f"{name}: no file named {base}[.tab|.csv] in {source}")

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=None, help="download folder (default: same as --out)")
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--copy", action="store_true", help="copy already-extracted files instead of linking them")
    args = parser.parse_args()

    source = args.out if args.source is None else args.source
    problems = prepare(source, args.out, copy=args.copy)
    if problems:
        print("\nThe data folder is not complete:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    print(f"\n{args.out} is ready.")


if __name__ == "__main__":
    main()
