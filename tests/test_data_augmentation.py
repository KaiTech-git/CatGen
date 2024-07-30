##DATA AUGMENTATION TEST##
import pytest
import os
import src.data_augmentation as da
from pandas import DataFrame
from PIL import Image
import cv2
import numpy as np

folder_path_zoom = "tests/zoom/"
folder_path_augmentation = "tests/augmentation/"


def test_cv_img_load(tmpdir) -> None:
    """Test color channel conversion BGR -> RGB in function image_load."""
    # Create blue and red images 10x10
    blue_path = tmpdir + "/blue.jpg"
    red_path = tmpdir + "/red.jpg"
    img_PIL_blue = Image.new("RGB", (10, 10), color=(0, 0, 255))
    img_PIL_blue.save(blue_path)
    img_PIL_red = Image.new("RGB", (10, 10), color=(255, 0, 0))
    img_PIL_red.save(red_path)
    assert os.path.exists(blue_path)
    assert os.path.exists(red_path)

    # Read images with cv2
    img_cv_blue = cv2.imread(blue_path)
    img_cv_red = cv2.imread(red_path)

    # Apply image_loaf func
    img_rev_blue = da.image_load(blue_path)
    img_rev_red = da.image_load(red_path)

    assert type(img_cv_red) == np.ndarray
    assert type(img_cv_blue) == np.ndarray

    assert (img_rev_blue[:, :, 2] & img_cv_blue[:, :, 0]).all()
    assert (img_rev_red[:, :, 0] & img_cv_red[:, :, 2]).all()

    blue_path.remove()
    red_path.remove()
    assert not blue_path.exists()
    assert not red_path.exists()


@pytest.mark.parametrize(
    "transformation",
    [
        ("No_Trans", da.no_transformation),
        ("Flip_Hor", da.horizontal_flip_transformation),
        ("Bright_Light", da.brightness_light_transformation),
        ("Bright_Dim", da.brightness_dim_transformation),
        ("Contrast", da.contrast_transformation),
        ("Satur", da.saturation_transformation),
        ("Post", da.posterize_transformation),
        ("Crop_Random", da.crop_transformation),
    ],
    ids=[
        "No_Trans",
        "Flip_Hor",
        "Bright_Light",
        "Bright_Dim",
        "Contrast",
        "Satur",
        "Post",
        "Crop_Random",
    ],
)
def test_apply_transform(transformation) -> None:
    """Test all augmentation transformations"""
    error_log_df: DataFrame = da.apply_transform(
        folder_path_zoom,
        folder_path_augmentation + transformation[0],
        transformation[1],
        transformation[0],
    )
    assert os.path.exists(folder_path_augmentation + transformation[0])

    input_img = [file for file in os.listdir(folder_path_zoom) if file[-5:] == ".jpeg"]
    out_img = [
        file
        for file in os.listdir(folder_path_augmentation + transformation[0])
        if file[-5:] == ".jpeg"
    ]

    err_num = sum(error_log_df)
    print(f"{error_log_df=}")
    assert len(input_img) == len(out_img) + err_num
