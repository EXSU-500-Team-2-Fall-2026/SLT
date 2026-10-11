"""Download HAM10000 (and the associated ISIC 2018 Task 3 test set / reader
study comparison files) from the Harvard Dataverse record cited in README.md:

    doi:10.7910/DVN/DBW86T  (https://doi.org/10.7910/DVN/DBW86T)

The dataset must not be committed to this repository (see Section 5.4 of
the course specification), so this script is how anyone reproduces the
download instead.

Usage:
    python -m src.data.download --list              # see what's in the record
    python -m src.data.download --out data/raw       # download the files we need
    python -m src.data.download --out data/raw --all # download everything in the record
"""
import argparse
import json
import re
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

DOI = "doi:10.7910/DVN/DBW86T"
API_BASE = "https://dataverse.harvard.edu/api"

# Harvard Dataverse returns 403 Forbidden to requests with Python's default
# urllib User-Agent (likely basic bot filtering). A normal browser/CLI UA
# string is enough to get through.
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) ham10000-download-script/1.0"

# Files are matched by (lowercased) name against these patterns. The exact
# file names on the Dataverse record were not re-verified when this script
# was written (no network access at the time) -- run with --list first and
# adjust this list if nothing matches.
DEFAULT_KEEP_PATTERNS = [
    r"ham10000.*images",
    r"ham10000.*metadata",
    r"ham10000.*segmentation",
    r"isic2018.*task3.*test",
    r"naturemedicine",
]


def list_files() -> list[dict]:
    url = f"{API_BASE}/datasets/:persistentId/?persistentId={DOI}"
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=30) as resp:
        payload = json.load(resp)
    files = payload["data"]["latestVersion"]["files"]
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
    url = f"{API_BASE}/access/datafile/{file_id}"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=300) as resp, open(dest_path, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)


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
    except URLError as exc:
        sys.exit(
            f"Could not reach the Harvard Dataverse API ({exc}).\n"
            "Check your network access to dataverse.harvard.edu, or download "
            "the files manually from https://doi.org/10.7910/DVN/DBW86T and "
            f"place them under {args.out}."
        )

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
        download_file(f["id"], dest)

    print(f"\nDone. Unzip any .zip archives under {args.out} before running make_splits.py.")


if __name__ == "__main__":
    main()
