# %%Imports
import os

import src.data_augmentation as da
from src.data_cleaning import find_duplicates_hash
import src.post_augmentation as pa
import model_arch as m
import torch as t

from torch.utils.data import DataLoader
import pandas as pd
from IPython.display import display

device = t.device("mps" if t.cuda.is_available() else "cpu")
print(device)
# TODO setup folders

# %% Variables
original_data = "data/data_with_background/0.ORIGINAL"
no_dup_data = "data/data_with_background/1.No_duplicates"
augmented_data = "data/data_with_background/2.Augmented_data"
crop_resize_folder = "data/data_with_background/3.Photos_64px"
# %% Remove Duplicates From Original
# Move data to new folder without transformations
da.apply_transform(original_data, no_dup_data, da.no_transformation, "cat")

# Create DataFrame with duplicated images
duplicates_df = find_duplicates_hash(no_dup_data)
print("List of duplicated images:")
duplicates_df
# %%
# Remove every other file from the list
dup_iter = iter(duplicates_df["file"])
img_deleted = []
for dup in dup_iter:
    dup_path = os.path.join(no_dup_data, dup)
    next(dup_iter)
    os.remove(dup_path)
    img_deleted.append(dup)
print(f"Number of duplicates deleted {len(img_deleted)}")


# %% Data Augmentation
# Crete list of all transformations
transformations_dict = {
    "Flip": da.horizontal_flip_transformation,
    "Bright": da.brightness_transformation,
    "Contrast": da.contrast_transformation,
    "Satur": da.saturation_transformation,
    "Post": da.posterize_transformation,
    "Rand_crop": da.crop_transformation,
}

# Move data to new folder without transformations
da.apply_transform(no_dup_data, augmented_data, da.no_transformation, "cat")
files_in_folder = len(os.listdir(augmented_data))
print(f"Initial number of img {files_in_folder}")
# Perform transformations
for name, trans in transformations_dict.items():
    error_log_df = da.apply_transform(
        augmented_data, augmented_data, trans, f"cat{name}"
    )
    files_in_folder = len(os.listdir(augmented_data))
    print(f"Number of imgs after {name} transformation: {files_in_folder}")
print(f"DATA AUGMENTATION ERROR LOG:")
display(error_log_df)

# %%Crop and resize photos to 64px by 64px
pa.crop_resize(augmented_data, crop_resize_folder, 64)
pa.get_random_img(crop_resize_folder, 10)
# %% Load dataset
train_set = m.get_dataset(crop_resize_folder)
x = next(iter(DataLoader(train_set, batch_size=64)))[0]
m.display_data(x, 8, "Loaded img")


