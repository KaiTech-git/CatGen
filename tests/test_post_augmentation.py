##POST AUGMENTATION TEST##
import pytest
import os
import src.post_augmentation as pa
from pandas import DataFrame
from PIL import Image


# Test performed on "tests/augmentation/Flip_Hor/""
folder_path_augmentation = "tests/augmentation/Flip_Hor/"


@pytest.mark.parametrize(
    "conditions",
    [
        {"path": "tests/post_augmentation/expand64/", "pix": 64, "keep": True},
        {"path": "tests/post_augmentation/crop64/", "pix": 64, "keep": False},
        {"path": "tests/post_augmentation/expand128/", "pix": 128, "keep": True},
        {"path": "tests/post_augmentation/crop128/", "pix": 128, "keep": False},
    ],
    ids=["Expand64", "Crop64", "Expand128", "Crop128"],
)
def test_expand_resize(conditions) -> None:
    pa.crop_resize(
        folder_path_augmentation,
        conditions["path"],
        conditions["pix"],
        conditions["keep"],
    )
    assert os.path.exists(conditions["path"])

    input_img = [
        file for file in os.listdir(folder_path_augmentation) if file[-5:] == ".jpeg"
    ]
    out_img = [file for file in os.listdir(conditions["path"]) if file[-5:] == ".jpeg"]

    assert len(input_img) == len(out_img)
    final_size = [Image.open(conditions["path"] + img).size for img in out_img]
    print(f"{final_size=}")
    assert all(([el == (conditions["pix"], conditions["pix"]) for el in final_size]))
