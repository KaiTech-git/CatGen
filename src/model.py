# %% Import
from typing import Literal, List
from torch.utils.data import Dataset
import torch as t
import einops
from torchvision import transforms
import plotly.express as px
import pathlib
from PIL import Image
from einops.layers.torch import Rearrange
from dataclasses import dataclass, field
from typing import Tuple, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm

device = t.device("cuda" if t.cuda.is_available() else "cpu")


# %% Create Custom Dataset Class
# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):

    # 2. Initialize with a targ_dir parameter
    def __init__(self, targ_dir: str, transform=None) -> None:

        # 3. Create class attributes
        # Get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*.jpeg"))
        print(self.paths)
        # Setup transforms
        self.transform = transform

        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = ["cat"], {"cat": 1}  # find_classes(targ_dir)

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)

    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> t.Tensor:
        "Returns one sample of data (X,y)."
        img = self.load_image(index)
        class_name = "cat"  # self.paths[index].parent.name  expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx  # return data
        else:
            return img, class_idx  # return data


# TODO Create tests for dataset
# train_data_custom = ImageFolderCustom(targ_dir="data/data_with_background/3.Photos_64px")
# len(train_data_custom)
# %% Loading Data
def get_dataset(dataset_path) -> Dataset:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_set = ImageFolderCustom(targ_dir=dataset_path, transform=transform)
    return train_set


def display_data(x: t.Tensor, nrows: int, title: str):
    """Displays a batch of data, using plotly."""
    # Reshape into the right shape for plotting
    y = einops.rearrange(x, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=nrows).squeeze()
    # Normalize, in the 0-1 range
    y = (y - y.min()) / (y.max() - y.min())
    # Display data
    fig = px.imshow(
        y,
        binary_string=(y.ndim == 2),
        height=50 * (nrows + 5),
        title=title + f"<br>single input shape = {x[0].shape}",
    )
    fig.show()
