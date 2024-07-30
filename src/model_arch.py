# %% Import
from typing import Literal, List
from torch.utils.data import Dataset
import torch as t
import einops
from torchvision import datasets, transforms
import plotly.express as px
import os
import pathlib
from PIL import Image
from einops.layers.torch import Rearrange
import torchinfo
from dataclasses import dataclass, field
from typing import Tuple, Optional
import wandb
from torch.utils.data import DataLoader
import time
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


# %% Define Model DCGAN


class Generator(t.nn.Module):
    def __init__(
        self,
        latent_dim_size: int = 100,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: List[int] = [128, 256, 512],
    ):

        n_layers = len(hidden_channels)
        assert (
            img_size % (2**n_layers) == 0
        ), "activation size must double at each layer"
        super().__init__()
        # Reverse hidden channels, so they're in chronological order
        hidden_channels = hidden_channels[::-1]

        self.latent_dim_size = latent_dim_size
        self.img_size = img_size
        self.img_channels = img_channels
        # Reverse them, so they're in chronological order for generator
        self.hidden_channels = hidden_channels

        # Define the first layer, i.e. latent dim -> (512, 4, 4) and reshape
        first_height = img_size // (2**n_layers)
        first_size = hidden_channels[0] * (first_height**2)
        self.project_and_reshape = t.nn.Sequential(
            t.nn.Linear(latent_dim_size, first_size, bias=False),
            Rearrange("b (ic h w) -> b ic h w", h=first_height, w=first_height),
            t.nn.BatchNorm2d(hidden_channels[0]),
            t.nn.ReLU(),
        )

        # Get list of input & output channels for the convolutional blocks
        in_channels = hidden_channels
        out_channels = hidden_channels[1:] + [img_channels]

        # Define all the convolutional blocks (conv_transposed -> batchnorm -> activation)
        conv_layer_list = []
        for i, (c_in, c_out) in enumerate(zip(in_channels, out_channels)):
            conv_layer = [
                t.nn.ConvTranspose2d(c_in, c_out, 4, 2, 1),
                t.nn.ReLU() if i < n_layers - 1 else t.nn.Tanh(),
            ]
            if i < n_layers - 1:
                conv_layer.insert(1, t.nn.BatchNorm2d(c_out))
            conv_layer_list.append(t.nn.Sequential(*conv_layer))

        self.hidden_layers = t.nn.Sequential(*conv_layer_list)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.project_and_reshape(x)
        x = self.hidden_layers(x)
        return x


class Discriminator(t.nn.Module):

    def __init__(
        self,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: List[int] = [128, 256, 512],
    ):

        n_layers = len(hidden_channels)
        assert (
            img_size % (2**n_layers) == 0
        ), "activation size must double at each layer"

        super().__init__()

        self.img_size = img_size
        self.img_channels = img_channels
        self.hidden_channels = hidden_channels

        # Get list of input & output channels for the convolutional blocks
        in_channels = [img_channels] + hidden_channels[:-1]
        out_channels = hidden_channels

        # Define all the convolutional blocks (conv_transposed -> batchnorm -> activation)
        conv_layer_list = []
        for i, (c_in, c_out) in enumerate(zip(in_channels, out_channels)):
            conv_layer = [
                t.nn.Conv2d(c_in, c_out, 4, 2, 1),
                t.nn.LeakyReLU(0.2),
            ]
            if i > 0:
                conv_layer.insert(1, t.nn.BatchNorm2d(c_out))
            conv_layer_list.append(t.nn.Sequential(*conv_layer))

        self.hidden_layers = t.nn.Sequential(*conv_layer_list)

        # Define the last layer, i.e. reshape and (512, 4, 4) -> real/fake classification
        final_height = img_size // (2**n_layers)
        final_size = hidden_channels[-1] * (final_height**2)
        self.classifier = t.nn.Sequential(
            Rearrange("b c h w -> b (c h w)"),
            t.nn.Linear(final_size, 1, bias=False),
            t.nn.Sigmoid(),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.hidden_layers(x)
        x = self.classifier(x)
        return x.squeeze()  # remove dummy out_channels dimension


class DCGAN(t.nn.Module):
    netD: Discriminator
    netG: Generator

    def __init__(
        self,
        latent_dim_size: int = 100,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: List[int] = [128, 256, 512],
    ):
        super().__init__()
        self.latent_dim_size = latent_dim_size
        self.img_size = img_size
        self.img_channels = img_channels
        self.hidden_channels = hidden_channels
        self.netD = Discriminator(img_size, img_channels, hidden_channels)
        self.netG = Generator(latent_dim_size, img_size, img_channels, hidden_channels)
        # initialize_weights(self)


# %% Check Model Parameters
if __name__ == "__main__":
    model = DCGAN().to(device)
    x = t.randn(3, 100).to(device)
    statsG = torchinfo.summary(model.netG, input_data=x)
    statsD = torchinfo.summary(model.netD, input_data=model.netG(x))
    print(statsG, statsD)


# %% Model Training Loop
@dataclass
class DCGANArgs:
    """
    Class for the arguments to the DCGAN (training and architecture).
    Note, we use field(defaultfactory(...)) when our default value is a mutable object.
    """

    latent_dim_size: int = 100
    hidden_channels: List[int] = field(default_factory=lambda: [128, 256, 512])
    # dataset: Literal["MNIST", "CELEB"] = "CELEB"
    dataset_path: str = "data/3.Photos_64px"
    batch_size: int = 64
    epochs: int = 3
    lr: float = 0.0002
    betas: Tuple[float] = (0.5, 0.999)
    seconds_between_eval: int = 20
    wandb_project: Optional[str] = "CatGen64"
    wandb_name: Optional[str] = None


class DCGANTrainer:
    def __init__(self, args: DCGANArgs):
        self.args = args
        # self.criterion = t.nn.BCELoss()

        self.trainset = get_dataset(self.args.dataset_path)
        self.trainloader = DataLoader(
            self.trainset, batch_size=args.batch_size, shuffle=True
        )

        batch, img_channels, img_height, img_width = next(iter(self.trainloader))[
            0
        ].shape
        assert img_height == img_width

        self.model = (
            DCGAN(
                args.latent_dim_size,
                img_height,
                img_channels,
                args.hidden_channels,
            )
            .to(device)
            .train()
        )

        self.optG = t.optim.Adam(
            self.model.netG.parameters(), lr=args.lr, betas=args.betas
        )
        self.optD = t.optim.Adam(
            self.model.netD.parameters(), lr=args.lr, betas=args.betas
        )

    def training_step_discriminator(
        self, img_real: t.Tensor, img_fake: t.Tensor
    ) -> t.Tensor:
        """
        Generates a real and fake image, and performs a gradient step on the discriminator
        to maximize log(D(x)) + log(1-D(G(z))).
        """
        # Zero gradients
        self.optD.zero_grad()

        # Calculate D(x) and D(G(z)), for use in the objective function
        D_x = self.model.netD(img_real)
        D_G_z = self.model.netD(img_fake)

        # Calculate loss
        lossD = -(t.log(D_x).mean() + t.log(1 - D_G_z).mean())

        # Gradient descent step
        lossD.backward()
        self.optD.step()

        return lossD

    def training_step_generator(self, img_fake: t.Tensor) -> t.Tensor:
        """
        Performs a gradient step on the generator to maximize log(D(G(z))).
        """
        # Zero gradients
        self.optG.zero_grad()

        # Calculate D(G(z)), for use in the objective function
        D_G_z = self.model.netD(img_fake)

        # Calculate loss
        lossG = -(t.log(D_G_z).mean())

        # Gradient descent step
        lossG.backward()
        self.optG.step()

        return lossG

    @t.inference_mode()
    def evaluate(self) -> None:
        """
        Performs evaluation by generating 8 instances of random noise and passing them through
        the generator, then logging the results to Weights & Biases.
        """
        self.model.netG.eval()

        # Generate random noise
        t.manual_seed(42)
        noise = t.randn(8, self.model.latent_dim_size).to(device)
        # Get generator output, turn it into an array
        arrays = (
            einops.rearrange(self.model.netG(noise), "b c h w -> b h w c").cpu().numpy()
        )
        # Log to weights and biases
        wandb.log({"images": [wandb.Image(arr) for arr in arrays]}, step=self.step)

        self.model.netG.train()

    def train(self) -> None:
        """
        Performs a full training run, while logging to Weights & Biases.
        """
        self.step = 0
        last_log_time = time.time()
        wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)

        for epoch in range(self.args.epochs):

            progress_bar = tqdm(self.trainloader, total=len(self.trainloader))

            for img_real, label in progress_bar:

                # Generate random noise & fake image
                noise = t.randn(self.args.batch_size, self.args.latent_dim_size).to(
                    device
                )
                img_real = img_real.to(device)
                img_fake = self.model.netG(noise)

                # Training steps
                lossD = self.training_step_discriminator(img_real, img_fake.detach())
                lossG = self.training_step_generator(img_fake)

                # Log data
                wandb.log(dict(lossD=lossD, lossG=lossG), step=self.step)

                # Update progress bar
                self.step += img_real.shape[0]
                progress_bar.set_description(
                    f"{epoch=}, lossD={lossD:.4f}, lossG={lossG:.4f}, examples_seen={self.step}"
                )

                # Evaluate model on the same batch of random data
                if time.time() - last_log_time > self.args.seconds_between_eval:
                    last_log_time = time.time()
                    self.evaluate()

        wandb.finish()


# %%

# %%
