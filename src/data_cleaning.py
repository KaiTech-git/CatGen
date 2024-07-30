# %% Imports
import hashlib
import pandas as pd
import os
from src.data_augmentation import apply_transform, no_transformation
from rembg import remove
import numpy as np
from PIL import Image, ImageChops
from tqdm import tqdm


###DATA CLEANING MODULE###
# %% FUNCTIONS TO LOOK FOR DUPLICATES
def calc_hash(file_path: str, block_size: int = 8192) -> str:
    """Generate the hash of any image according to MD5 algorithm.
    Hashing function to find duplicates in original data
    Function implemented based on this article
    https://towardsdatascience.com/find-duplicate-photos-and-other-files-88b0d07ef020
    This function identifies only exactly the same photos.
    If to photos are the same but won't have the same size function won't detect them.
    """

    file_hash = hashlib.md5()  # Define hashing algorithm
    with open(file_path, "rb") as message:  # Open file in binary mode
        while block := message.read(block_size):
            file_hash.update(block)  # Update hash by next block of bites
    return file_hash.hexdigest()  # Return hashing of the file


def find_duplicates_hash(folder_path: str) -> pd.DataFrame:
    """Create a pandas DataFrame with identified duplicated photos."""

    # Get all jpeg images in directory
    file_list = [file for file in os.listdir(folder_path) if file.endswith(".jpeg")]
    path_list = [os.path.join(folder_path, f) for f in file_list]

    # Crate DF with file names and corresponding hash representation
    duplicates_df = pd.DataFrame(columns=["file", "hash"])
    duplicates_df["file"] = file_list
    duplicates_df["hash"] = [calc_hash(f) for f in path_list]

    # Create DF with duplicated photos
    duplicates_df = duplicates_df[duplicates_df["hash"].duplicated(keep=False)]
    duplicates_df = duplicates_df.sort_values(by=["hash"])

    return duplicates_df


def rem_duplicates(file_path: str, dup_list: pd.DataFrame) -> int:
    """Remove duplicated photos identified by find_duplicates_hash."""
    seen_img = set()
    for dup_f, dup_h in dup_list.itertuples(index=False):
        if dup_h in seen_img:
            os.remove(os.path.join(file_path, dup_f))
        else:
            seen_img.add(dup_h)
    return dup_list.shape[0] - len(seen_img)


# %% REMOVE BACKGROUND FROM THE PHOTO
def rem_bg(input_path: str, output_path: str):
    """Remove background from original photos using rembg lib."""
    # Create output folder if needed
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file_list = os.listdir(input_path)
    file_list.sort()
    # Iterate over input photos
    for i, img in tqdm(enumerate(file_list), total=len(file_list)):
        # Load the image
        try:
            img_path = os.path.join(input_path, img)
            image = Image.open(img_path)

            # Convert the input image to a numpy array
            input_array = np.array(image)

            # Apply background removal using rembg
            output_array = remove(
                input_array,
                bgcolor=(255, 255, 255, 255),
                alpha_matting_foreground_threshold=270,
                alpha_matting_background_threshold=20,
                alpha_matting_erode_size=11,
            )

            # Create a PIL Image from the output array
            output_image = Image.fromarray(output_array).convert("RGB")

            # Save the output image
            save_path = os.path.join(output_path, f"{i}_cat_bg.jpeg")
            output_image.save(save_path)
        except Image.UnidentifiedImageError as err:
            print(f"Error {err} occurred with file {img}")


# %% ZOOM ON IMAGE
def zoom_cat(input_path: str, output_path: str):
    """Zoom in on object after background was removed."""
    # Create output folder if needed
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file_list = os.listdir(input_path)
    file_list.sort()
    # Iterate over input photos
    for i, file in tqdm(enumerate(file_list), total=len(file_list)):
        # Load the image
        try:
            img_path = os.path.join(input_path, file)
            img = Image.open(img_path)

            # Create bg image with the color of top-left pixel
            bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
            # Computes the pixel-by-pixel difference between img and bg
            diff = ImageChops.difference(img, bg)
            # Enhances the differences and add offset -100 to darken the non-differing regions
            diff = ImageChops.add(diff, diff, 2.0, -100)
            # Calculates the bounding box of the non-zero regions
            bbox = diff.getbbox()
            # Zoom if bounding box is not None
            if bbox:
                output_image = img.crop(bbox)

            else:
                output_image = img
            # Save the output image
            save_path = os.path.join(output_path, f"{i}_cat_zoom.jpeg")
            output_image.save(save_path)
        except Image.UnidentifiedImageError as err:
            print(f"Error {err} occurred with file {img}")


# %% Create folder without duplicates
if __name__ == "__main__":
    input_file = "Photos"
    no_dup_file = "Photos_no_duplicates"
    apply_transform(input_file, no_dup_file, no_transformation, "kota")
    duplicates_df = find_duplicates_hash(no_dup_file)
    img_removed = rem_duplicates(no_dup_file, duplicates_df)
    print(f"Number of duplicates deleted {img_removed}")
