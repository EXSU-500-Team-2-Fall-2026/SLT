"""Download script, tested without any network access by replacing
requests.get with a stand-in."""
import pytest
import requests

from src.data import download as dl

# File names as they appear in a download of the Dataverse record.
RECORD_FILES = [
    "HAM10000_images_part_1.zip",
    "HAM10000_images_part_2.zip",
    "HAM10000_metadata.tab",
    "HAM10000_segmentations_lesion_tschandl.zip",
    "ISIC2018_Task3_Test_Images.zip",
    "ISIC2018_Task3_Test_GroundTruth.tab",
    "ISIC2018_Task3_Test_NatureMedicine_AI_Interaction_Benefit.tab",
]


class FakeResponse:
    def __init__(self, payload=None, body=b"", fail=False):
        self.payload, self.body, self.fail = payload, body, fail

    def raise_for_status(self):
        if self.fail:
            raise requests.HTTPError("403 Forbidden")

    def json(self):
        return self.payload

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), 4):
            yield self.body[i:i + 4]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _listing():
    return {"data": {"latestVersion": {"files": [
        {"dataFile": {"id": i + 1, "filename": name, "filesize": 1000 * (i + 1)}}
        for i, name in enumerate(RECORD_FILES + ["README.txt"])
    ]}}}


@pytest.fixture()
def fake_api(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        if "/access/datafile/" in url:
            return FakeResponse(body=b"file-contents")
        return FakeResponse(payload=_listing())

    monkeypatch.setattr(dl.requests, "get", fake_get)
    return calls


def test_list_files_parses_the_record(fake_api):
    files = dl.list_files()
    assert [f["name"] for f in files] == RECORD_FILES + ["README.txt"]
    assert files[0] == {"id": 1, "name": "HAM10000_images_part_1.zip", "size": 1000}
    url, kwargs = fake_api[0]
    assert kwargs["params"] == {"persistentId": dl.DOI}
    assert kwargs["headers"]["User-Agent"] == dl.USER_AGENT


@pytest.mark.parametrize("name", RECORD_FILES + ["HAM10000_metadata", "ISIC2018_Task3_Test_GroundTruth.csv"])
def test_keep_patterns_match_every_needed_file(name):
    assert dl.matches_any(name, dl.DEFAULT_KEEP_PATTERNS)


def test_keep_patterns_skip_unrelated_files():
    assert not dl.matches_any("README.txt", dl.DEFAULT_KEEP_PATTERNS)


def test_download_file_writes_complete_file_and_no_leftover(fake_api, tmp_path):
    dest = tmp_path / "raw" / "HAM10000_metadata.tab"
    dl.download_file(3, dest)
    assert dest.read_bytes() == b"file-contents"
    assert [p.name for p in dest.parent.iterdir()] == ["HAM10000_metadata.tab"]


def test_failed_download_leaves_no_finished_file(monkeypatch, tmp_path):
    monkeypatch.setattr(dl.requests, "get", lambda url, **kw: FakeResponse(fail=True))
    dest = tmp_path / "HAM10000_metadata.tab"
    with pytest.raises(requests.HTTPError):
        dl.download_file(3, dest)
    assert not dest.exists()


def test_main_downloads_needed_files_and_skips_existing(fake_api, tmp_path, monkeypatch, capsys):
    out = tmp_path / "raw"
    out.mkdir()
    (out / "HAM10000_metadata.tab").write_bytes(b"already here")
    monkeypatch.setattr("sys.argv", ["download", "--out", str(out)])
    dl.main()

    assert sorted(p.name for p in out.iterdir()) == sorted(RECORD_FILES)
    assert (out / "HAM10000_metadata.tab").read_bytes() == b"already here"
    assert "prepare_raw" in capsys.readouterr().out


def test_main_list_downloads_nothing(fake_api, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["download", "--out", str(tmp_path / "raw"), "--list"])
    dl.main()
    assert not (tmp_path / "raw").exists()
    assert "HAM10000_images_part_1.zip" in capsys.readouterr().out


def test_network_failure_gives_a_clear_message(monkeypatch, tmp_path):
    def broken(url, **kwargs):
        raise requests.exceptions.SSLError("certificate verify failed")

    monkeypatch.setattr(dl.requests, "get", broken)
    monkeypatch.setattr("sys.argv", ["download", "--out", str(tmp_path / "raw"), "--list"])
    with pytest.raises(SystemExit) as exit_info:
        dl.main()
    message = str(exit_info.value)
    assert "Could not get the files" in message and "prepare_raw --source" in message
