# %% Imports
import os
import cv2
import plotly.express as px
from PIL import Image
import numpy as np
import random


###POST AUGMENTATION MODULE##
# %%CROP OR EXPAND IMAGES AND RESIZE TO MODEL SIZE
def crop_resize(
    input_folder: str, output_folder: str, px: int = 64, keep: bool = False
) -> None:
    # Create output file if doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for n, file_name in enumerate(os.listdir(input_folder)):
        img_path = os.path.join(input_folder, file_name)
        try:
            with Image.open(img_path) as img:
                # If keep == True expend image not to lose any pixel
                # Image will be passed on to white square of dimensions square_dimension
                if keep:
                    square_dimension = max(img.width, img.height)
                    square_image = Image.new(
                        "RGB", (square_dimension, square_dimension), (255, 255, 255)
                    )
                    top_y = (square_dimension - img.height) // 2
                    top_x = (square_dimension - img.width) // 2
                    square_image.paste(img, (top_x, top_y))

                # If keep == False crop image to square of dimensions square_dimension
                else:
                    square_dimension = min(img.width, img.height)
                    left = (img.width - square_dimension) / 2
                    right = (img.width + square_dimension) / 2
                    top = (img.height - square_dimension) / 2
                    down = (img.height + square_dimension) / 2
                    square_image = img.crop((left, top, right, down))
                # Resize image
                img_resized = square_image.resize((px, px))

                # Save image
                output_path = os.path.join(output_folder, f"kota_{px}_{n}.jpeg")
                img_resized.save(output_path)
        except Image.UnidentifiedImageError as err:
            print(f"Error {err} occurred with file {file_name}")


# %% Function to print some examples of generated photos
def get_random_img(
    folder_path: str, n_samples: int, img_dims: tuple = (64, 64, 3)
) -> None:
    img_list = []
    try:
        selected_img = random.choices(os.listdir(folder_path), k=n_samples)
        for filename in selected_img:
            file_path = os.path.join(folder_path, filename)
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            try:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except:
                print("Error occurred when opencv was opening the file.")
                img = np.zeros(img_dims)
            img_list.append(img)
        img_hor = cv2.hconcat(img_list)
        fig = px.imshow(img_hor)
        fig.show()
    except FileNotFoundError as err:
        print(f"Error {err} make sure that folder is in current dir.")


# %% Crop and resize images to 64 x 64px
if __name__ == "__main__":
    input_folder = "Photos"
    crop_resize_folder = "photos_64"
    crop_resize(input_folder, crop_resize_folder, 64)
    get_random_img(crop_resize_folder, 5)
