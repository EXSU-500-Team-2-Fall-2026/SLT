"""Download HAM10000 (and the associated ISIC 2018 Task 3 test set / reader
study comparison files) from the Harvard Dataverse record cited in README.md:

    doi:10.7910/DVN/DBW86T  (https://doi.org/10.7910/DVN/DBW86T)

The dataset must not be committed to this repository (see Section 5.4 of
the course specification), so this script is how anyone reproduces the
download instead. After downloading, run src/data/prepare_raw.py to
arrange the files into the layout the code expects.

Usage:
    python -m src.data.download --list              # see what's in the record
    python -m src.data.download --out data/raw       # download the files we need
    python -m src.data.download --out data/raw --all # download everything in the record
"""
import argparse
import re
import sys
from pathlib import Path

import requests

DOI = "doi:10.7910/DVN/DBW86T"
API_BASE = "https://dataverse.harvard.edu/api"

# Harvard Dataverse returns 403 Forbidden to requests with a default Python
# User-Agent (likely basic bot filtering). A normal browser/CLI UA string
# is enough to get through.
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) ham10000-download-script/1.0"
HEADERS = {"User-Agent": USER_AGENT}

# The requests library is used rather than urllib because it checks HTTPS
# certificates against its own bundled list (certifi). urllib relies on the
# Python installation's certificate store, which is empty on python.org
# builds for macOS until "Install Certificates.command" has been run, and
# the download then fails with CERTIFICATE_VERIFY_FAILED.

# Files are matched by (lowercased) name against these patterns. Run with
# --list to see the record's current file names.
DEFAULT_KEEP_PATTERNS = [
    r"ham10000.*images",
    r"ham10000.*metadata",
    r"ham10000.*segmentation",
    r"isic2018.*task3.*test",
    r"naturemedicine",
]


def list_files() -> list[dict]:
    url = f"{API_BASE}/datasets/:persistentId/"
    resp = requests.get(url, params={"persistentId": DOI}, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    files = resp.json()["data"]["latestVersion"]["files"]
    return [
        {
            "id": f["dataFile"]["id"],
            "name": f["dataFile"]["filename"],
            "size": f["dataFile"].get("filesize"),
        }
        for f in files
    ]


def matches_any(name: str, patterns: list[str]) -> bool:
    name_l = name.lower()
    return any(re.search(p, name_l) for p in patterns)


def download_file(file_id: int, dest_path: Path) -> None:
    """Download to '<name>.part' and rename when complete, so an interrupted
    download is never mistaken for a finished file on the next run."""
    url = f"{API_BASE}/access/datafile/{file_id}"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_name(dest_path.name + ".part")
    with requests.get(url, headers=HEADERS, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(part_path, "wb") as out:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                out.write(chunk)
    part_path.replace(dest_path)


def _unreachable(exc: Exception, out: Path) -> str:
    return (
        f"Could not get the files from the Harvard Dataverse API ({exc}).\n"
        "Check your network access to dataverse.harvard.edu, or download the "
        "files in a browser from https://doi.org/10.7910/DVN/DBW86T and then run\n"
        f"    python -m src.data.prepare_raw --source <download folder> --out {out}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--list", action="store_true", help="list files and exit")
    parser.add_argument(
        "--all", action="store_true", help="download every file, ignoring keep-patterns"
    )
    args = parser.parse_args()

    try:
        files = list_files()
    except requests.RequestException as exc:
        sys.exit(_unreachable(exc, args.out))

    print(f"{len(files)} files in dataset {DOI}:")
    for f in files:
        size_mb = (f["size"] or 0) / 1e6
        print(f"  [{f['id']:>8}] {f['name']:<50} {size_mb:7.1f} MB")

    if args.list:
        return

    to_get = (
        files if args.all else [f for f in files if matches_any(f["name"], DEFAULT_KEEP_PATTERNS)]
    )
    if not to_get:
        sys.exit(
            "No files matched the keep-patterns. Run with --list to see the "
            "actual file names and update DEFAULT_KEEP_PATTERNS in this script."
        )

    for f in to_get:
        dest = args.out / f["name"]
        if dest.exists():
            print(f"skip (already exists): {dest}")
            continue
        print(f"downloading {f['name']} -> {dest}")
        try:
            download_file(f["id"], dest)
        except requests.RequestException as exc:
            sys.exit(_unreachable(exc, args.out))

    print(f"\nDone. Next: python -m src.data.prepare_raw --out {args.out}")


if __name__ == "__main__":
    main()
