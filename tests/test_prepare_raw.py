"""prepare_raw on small fake downloads that copy the structure of the real
Dataverse archives (flat training zips, macOS junk in the others)."""
import zipfile

import pytest

from src.data import prepare_raw as pr

EXPECTED = {"HAM10000_images": 5, "ISIC2018_Task3_Test_Images": 2, "HAM10000_segmentations_lesion_tschandl": 5}
TRAIN_1 = ["ISIC_0000001.jpg", "ISIC_0000002.jpg", "ISIC_0000003.jpg"]
TRAIN_2 = ["ISIC_0000004.jpg", "ISIC_0000005.jpg"]
TEST = ["ISIC_0009001.jpg", "ISIC_0009002.jpg"]
MASKS = [name.replace(".jpg", "_segmentation.png") for name in TRAIN_1 + TRAIN_2]


def _zip(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name in members:
            zf.writestr(name, b"x")


def _zipped_download(folder):
    folder.mkdir()
    _zip(folder / "HAM10000_images_part_1.zip", TRAIN_1)
    _zip(folder / "HAM10000_images_part_2.zip", TRAIN_2)
    _zip(
        folder / "ISIC2018_Task3_Test_Images.zip",
        ["ISIC2018_Task3_Test_Images/"]
        + [f"ISIC2018_Task3_Test_Images/{n}" for n in TEST]
        + ["__MACOSX/", "__MACOSX/ISIC2018_Task3_Test_Images/"]
        + [f"__MACOSX/ISIC2018_Task3_Test_Images/._{n}" for n in TEST]
        + ["ISIC2018_Task3_Test_Images/.DS_Store"],
    )
    _zip(
        folder / "HAM10000_segmentations_lesion_tschandl.zip",
        [f"HAM10000_segmentations_lesion_tschandl/{n}" for n in MASKS]
        + ["__MACOSX/HAM10000_segmentations_lesion_tschandl/._ISIC_0000001_segmentation.png"],
    )
    (folder / "HAM10000_metadata").write_text("lesion_id,image_id,dx\n")
    (folder / "ISIC2018_Task3_Test_GroundTruth.csv").write_text("lesion_id,image_id,dx\n")
    return folder


def _extracted_download(folder):
    folder.mkdir()
    for sub, names in (("HAM10000_images_part_1", TRAIN_1), ("HAM10000_images_part_2", TRAIN_2),
                       ("ISIC2018_Task3_Test_Images", TEST), ("HAM10000_segmentations_lesion_tschandl", MASKS)):
        (folder / sub).mkdir()
        for name in names:
            (folder / sub / name).write_bytes(b"x")
    (folder / "ISIC2018_Task3_Test_Images" / ".DS_Store").write_bytes(b"x")
    (folder / "HAM10000_metadata.tab").write_text("lesion_id\timage_id\tdx\n")
    (folder / "ISIC2018_Task3_Test_GroundTruth.tab").write_text("lesion_id\timage_id\tdx\n")
    return folder


def _assert_layout(out):
    assert sorted(p.name for p in (out / "HAM10000_images").iterdir()) == sorted(TRAIN_1 + TRAIN_2)
    assert sorted(p.name for p in (out / "ISIC2018_Task3_Test_Images").iterdir()) == sorted(TEST)
    assert sorted(p.name for p in (out / "HAM10000_segmentations_lesion_tschandl").iterdir()) == sorted(MASKS)
    assert (out / "HAM10000_metadata.tab").read_text().startswith("lesion_id")
    assert (out / "ISIC2018_Task3_Test_GroundTruth.tab").read_text().startswith("lesion_id")
    assert not (out / "__MACOSX").exists()


def test_zipped_download_in_another_folder(tmp_path):
    source = _zipped_download(tmp_path / "dataverse_files")
    before = sorted(p.name for p in source.iterdir())
    out = tmp_path / "data" / "raw"

    assert pr.prepare(source, out, expected=EXPECTED) == []
    _assert_layout(out)
    assert sorted(p.name for p in source.iterdir()) == before  # download folder untouched


def test_zipped_download_prepared_in_place(tmp_path):
    raw = _zipped_download(tmp_path / "raw")
    assert pr.prepare(raw, raw, expected=EXPECTED) == []
    _assert_layout(raw)


@pytest.mark.parametrize("copy", [False, True])
def test_extracted_download_is_linked_or_copied(tmp_path, copy):
    source = _extracted_download(tmp_path / "dataverse_files")
    out = tmp_path / "raw"
    assert pr.prepare(source, out, copy=copy, expected=EXPECTED) == []
    _assert_layout(out)
    image = out / "HAM10000_images" / TRAIN_1[0]
    assert image.is_symlink() != copy
    assert image.read_bytes() == b"x"


def test_training_images_unzipped_loose_into_raw(tmp_path):
    raw = _extracted_download(tmp_path / "raw")
    for sub, names in (("HAM10000_images_part_1", TRAIN_1), ("HAM10000_images_part_2", TRAIN_2)):
        for name in names:
            (raw / sub / name).rename(raw / name)
        (raw / sub).rmdir()
    assert pr.prepare(raw, raw, expected=EXPECTED) == []
    _assert_layout_images_only = sorted(p.name for p in (raw / "HAM10000_images").iterdir())
    assert _assert_layout_images_only == sorted(TRAIN_1 + TRAIN_2)


def test_running_twice_changes_nothing(tmp_path):
    source = _zipped_download(tmp_path / "dataverse_files")
    out = tmp_path / "raw"
    pr.prepare(source, out, expected=EXPECTED)
    marker = out / "HAM10000_images" / TRAIN_1[0]
    marker.write_bytes(b"edited")
    assert pr.prepare(source, out, expected=EXPECTED) == []
    assert marker.read_bytes() == b"edited"  # existing files are not overwritten


def test_reports_missing_pieces(tmp_path):
    source = _zipped_download(tmp_path / "dataverse_files")
    (source / "HAM10000_images_part_2.zip").rename(tmp_path / "elsewhere.zip")
    (source / "HAM10000_metadata").rename(tmp_path / "elsewhere.txt")
    problems = pr.prepare(source, tmp_path / "raw", expected=EXPECTED)
    assert any("HAM10000_images" in p and "found 3" in p for p in problems)
    assert any("HAM10000_metadata.tab" in p for p in problems)


def test_missing_optional_masks_is_not_a_problem(tmp_path):
    source = _zipped_download(tmp_path / "dataverse_files")
    (source / "HAM10000_segmentations_lesion_tschandl.zip").rename(tmp_path / "elsewhere.zip")
    assert pr.prepare(source, tmp_path / "raw", expected=EXPECTED) == []


def test_missing_download_folder(tmp_path):
    problems = pr.prepare(tmp_path / "nope", tmp_path / "raw", expected=EXPECTED)
    assert problems and "not found" in problems[0]


@pytest.mark.parametrize(
    "name, junk",
    [("__MACOSX/a/._ISIC_1.jpg", True), ("folder/.DS_Store", True), ("._ISIC_1.jpg", True),
     ("ISIC2018_Task3_Test_Images/ISIC_1.jpg", False), ("ISIC_1.jpg", False)],
)
def test_is_junk(name, junk):
    assert pr.is_junk(name) is junk
