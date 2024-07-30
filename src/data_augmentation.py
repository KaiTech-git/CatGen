# %%
import cv2
import plotly.express as px
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import pandas as pd
from IPython.display import display
from tqdm import tqdm


###DATA AUGMENTATION###
"""This module was created based on the article from kaggle
https://www.kaggle.com/code/mohamedmustafa/7-data-augmentation-on-images-using-pytorch"""


# %%  DATA AUGMENTATION - PREP
def image_load(file_path: str) -> np.ndarray:
    """Cv2 when loading images represents colors in BGR format.
    Channels has to be converted in to RGB."""
    img = cv2.imread(file_path)[:, :, ::-1]  # Read image in BGR and convert to RGB

    return img


def show_img(img: np.ndarray, title: str) -> None:
    """Show image with plotly express"""
    fig = px.imshow(img, title=title)
    fig.show()


def error_log(error_log_dict: dict, err_type, new_file_name, img_path):
    """Transformations error log function"""
    error_log_dict["ErrorName"].append(err_type)
    error_log_dict["Transformation"].append(new_file_name)
    error_log_dict["FileName"].append(img_path)


# %% TRANSFORMATION DEFINITIONS
"""Transformations were defined based on article from pytorch:
https://pytorch.org/vision/0.13/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
For all the transformations probability of performing the transformation
was set to 1. The goal here is to get as many different photos 
as possible to augment original data. 
So of the transformations where applied to with background and some 
to without background training."""
# Change np.ndarray in to PILImage no image changes.
no_transformation = transforms.Compose([transforms.ToPILImage()])

# Flip img horizontally
horizontal_flip_transformation = transforms.Compose(
    [transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=1)]
)

# Brightness random change - brighten up
brightness_light_transformation = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=(1, 1.4), contrast=0, saturation=0, hue=0),
    ]
)

# Brightness random change - darken
brightness_dim_transformation = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=(0.7, 1), contrast=0, saturation=0, hue=0),
    ]
)

# Contrast random change
contrast_transformation = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0, contrast=0.3, saturation=0, hue=0),
    ]
)

# Saturation random change
saturation_transformation = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0.7, hue=0),
    ]
)

# Posterize random change
# Transformation reduces the number of bits of each color channel
posterize_transformation = transforms.Compose(
    [transforms.ToPILImage(), transforms.RandomPosterize(5, p=1)]
)

# Random crop to 240x180px
crop_transformation = transforms.Compose(
    [transforms.ToPILImage(), transforms.RandomCrop(size=(240, 180))]
)


# Apply transformation
def apply_transform(
    input_folder: str,
    output_folder: str,
    transformation: transforms.Compose,
    new_file_name: str,
) -> pd.DataFrame:
    # Create output file if doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    error_log_dict = {"ErrorName": [], "Transformation": [], "FileName": []}
    for n, file_name in tqdm(
        enumerate(os.listdir(input_folder)), total=len(os.listdir(input_folder))
    ):
        img_path = os.path.join(input_folder, file_name)
        try:
            img = image_load(img_path)
            img_transformed = transformation(img)

            # Save image
            output_path = os.path.join(output_folder, f"{n}_{new_file_name}.jpeg")
            img_transformed.save(output_path)
        except Image.UnidentifiedImageError as err:
            # print(f'Error {err} occurred with file {file_name}')
            error_log(error_log_dict, str(err), new_file_name, img_path)
        except TypeError as err:
            # print(f'Error {err} occurred with file {file_name}')
            error_log(error_log_dict, str(err), new_file_name, img_path)
        except ValueError as err:
            # print(f'Error {err} occurred with file {file_name}')
            error_log(error_log_dict, str(err), new_file_name, img_path)
    error_log_df = (
        pd.DataFrame(error_log_dict)
        .groupby(["ErrorName", "Transformation"])["FileName"]
        .count()
    )
    return error_log_df


# %% Data Augmentation - main
if __name__ == "__main__":
    #    error_log1 = apply_transform("data/data_with_background/0.ORIGINAL", "aug_photos",no_transformation,"kota")
    error_log2 = apply_transform(
        "data/data_with_background/0.ORIGINAL",
        "aug_photos",
        crop_transformation,
        "kota_flipped",
    )
    display(error_log2)
