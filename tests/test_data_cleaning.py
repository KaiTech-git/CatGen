##DATA CLEANING TEST##
import pytest
import os
import src.data_cleaning as dc
import src.data_augmentation as da
from typing import Tuple
from pandas import DataFrame
from PIL import Image


folder_path_dup = "tests/duplicate_photos/"
folder_path_rembg = "tests/rembg_photos/"
folder_path_zoom = "tests/zoom/"


def test_img_hash() -> None:
    """Test to check if calc_hash converts images to its hash representation."""
    img_hash = dc.calc_hash(folder_path_dup + "Test_cat1.jpeg")
    print(f"img hash: {img_hash}")


def img_path(file_name: str) -> str:
    """Crates relative path to files in tests/duplicate_photos folder"""
    return os.path.join("tests/duplicate_photos/", file_name)


"""Parameters in form of tuple where:
- first img is the file from original data
- second is copy of the first one
- third was also copied but later edited."""


@pytest.mark.parametrize(
    "img_compare",
    [
        ("Test_cat1.jpeg", "Test_cat1_dup1.jpeg", "Test_cat2_cut1.jpeg"),
        ("Test_cat3.jpeg", "Test_cat3_dup1.jpeg", "Test_cat3_resize1.jpeg"),
        ("Test_cat6.jpeg", "Test_cat6_dup2.jpeg", "Test_cat6_resize2.jpeg"),
    ],
    ids=["Cat1 Comparison", "Cat3 Comparison", "Cat6 Comparison"],
)
def test_duplicate_img(img_compare: Tuple[str]) -> None:
    """Test if calc_hash can identify duplicates."""

    # Identical photos
    assert dc.calc_hash(img_path(img_compare[0])) == dc.calc_hash(
        img_path(img_compare[1])
    )
    # Different photos
    assert dc.calc_hash(img_path(img_compare[0])) != dc.calc_hash(
        img_path(img_compare[2])
    )


def test_find_duplicates(folder_path_dup: str = folder_path_dup) -> None:
    """find_duplicates_hash function test
    Testing returned format and number of identified
    duplicates in duplicate photos folder."""

    dup_df = dc.find_duplicates_hash(folder_path_dup)

    assert type(dup_df) == DataFrame
    assert dup_df.shape[0] == 10


def test_find_jpeg_dup_only(tmpdir) -> None:
    """Test if dc.find_duplicates_hash identifies only jpeg duplicates"""

    # Create file path for text files
    tmp_dup_file1 = tmpdir.join("test_dup1.txt")
    tmp_dup_file2 = tmpdir.join("test_dup2.txt")
    tmp_dup_file3 = tmpdir.join("test_dup3.txt")
    tmp_diff_file4 = tmpdir.join("test_diff4.txt")

    # Write to files
    tmp_dup_file1.write("Contents of temp_file_dup")
    tmp_dup_file2.write("Contents of temp_file_dup")
    tmp_dup_file3.write("Contents of temp_file_dup")
    tmp_diff_file4.write("Contents of temp_file_diff")

    assert tmp_dup_file1.exists()
    assert tmp_dup_file2.exists()
    assert tmp_dup_file3.exists()
    assert tmp_diff_file4.exists()

    # Create file path for img files
    tmp_dup_img1 = tmpdir.join("test_dup1.jpeg")
    tmp_dup_img2 = tmpdir.join("test_dup2.jpeg")

    # Write content to img files
    img = Image.new("RGB", (100, 100), color="green")
    img.save(tmp_dup_img1)
    img.save(tmp_dup_img2)

    assert tmp_dup_img1.exists()
    assert tmp_dup_img2.exists()

    # Test dc.calc_hash
    assert dc.calc_hash(tmp_dup_file1) == dc.calc_hash(tmp_dup_file2)
    assert dc.calc_hash(tmp_dup_file1) != dc.calc_hash(tmp_diff_file4)
    assert dc.calc_hash(tmp_dup_img1) == dc.calc_hash(tmp_dup_img2)

    # Test if find_duplicates_hash identifies only jpeg duplicates
    dup_df = dc.find_duplicates_hash(tmpdir)
    assert dup_df.shape[0] == 2

    # Remove temporary files
    for file in tmpdir.listdir():
        file.remove()

    assert len(tmpdir.listdir()) == 0


def test_rem_duplicates(tmpdir) -> None:
    # Move data to new folder without transformations
    da.apply_transform(folder_path_dup, tmpdir, da.no_transformation, "cat")
    assert len(os.listdir(folder_path_dup)) == len(
        os.listdir(tmpdir)
    )  # all files copied

    print(dc.calc_hash(tmpdir.listdir()[0]))
    # Test if all the file are identical after transformation
    assert [dc.calc_hash(img) for img in tmpdir.listdir()].sort() == [
        dc.calc_hash(os.path.join(folder_path_dup + img))
        for img in os.listdir(folder_path_dup)
    ].sort()

    duplicates_df = dc.find_duplicates_hash(tmpdir)
    assert duplicates_df.shape[0] == 10  # number of identified duplicated photos

    rem_imgs = dc.rem_duplicates(tmpdir, duplicates_df)
    assert len(tmpdir.listdir()) == 9  # number of photos after deletion
    assert rem_imgs == 6  # number of deleted photos

    # Remove temporary files
    for file in tmpdir.listdir():
        file.remove()
    assert len(tmpdir.listdir()) == 0


def test_remove_background(tmpdir) -> None:
    """Test if background was removed part of this test has to be dan manually
    to check if quality of transformation is sufficient."""

    # Remove background
    dc.rem_bg(folder_path_rembg + "bg/", folder_path_rembg + "rem_bg/")
    assert os.path.exists(folder_path_rembg + "rem_bg/")
    # Get all the images from directory.
    input_img = [
        file for file in os.listdir(folder_path_rembg + "bg/") if file[-5:] == ".jpeg"
    ]
    out_img = [
        file
        for file in os.listdir(folder_path_rembg + "rem_bg/")
        if file[-5:] == ".jpeg"
    ]

    assert len(input_img) == len(out_img)


def test_zoom_cat() -> None:
    """Test Zoom function based on the assumption that image zoomed if it's size
    decried. It's also possible that the object entire photo in this case size
    won't change."""

    # Zoom images without background
    dc.zoom_cat(folder_path_rembg + "rem_bg/", folder_path_zoom)
    assert os.path.exists(folder_path_zoom)
    # Get all the images from directory.
    input_img = [
        file
        for file in os.listdir(folder_path_rembg + "rem_bg/")
        if file[-5:] == ".jpeg"
    ]
    out_img = [file for file in os.listdir(folder_path_zoom) if file[-5:] == ".jpeg"]
    assert len(input_img) == len(
        out_img
    )  # function generate the same number of images as input

    # Check if all the images where zoomed in
    input_img.sort()
    out_img.sort()
    compare_size = [
        Image.open(folder_path_rembg + "rem_bg/" + pare_0).size
        >= Image.open(folder_path_zoom + pare_1).size
        for pare_0, pare_1 in zip(input_img, out_img)
    ]
    assert sum(compare_size) == len(input_img)
