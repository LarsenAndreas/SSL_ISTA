from copy import deepcopy
from typing import Iterable

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor
from tqdm.auto import tqdm, trange
from utils import getDSMatrix
from model import ISTANet
from torch.nn.functional import fold, linear, relu, unfold, hardtanh, mse_loss
import numpy as np


class ImageData(Dataset):
    def __init__(self, paths: Iterable, crop_size: int = 32):
        """Extracts patches of `crop_size` from the images specified in `paths`.

        Args:
            `paths` (Iterable): List of paths to images.
            `crop_size` (int, optional): Size of the crop to be extracted. Defaults to 32.
        """
        self.paths = paths
        self.pre_process = Compose([ToTensor(), RandomCrop(size=(crop_size, crop_size))])

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as img:
            img = self.pre_process(img)
            return img.float()

    def __len__(self):
        return len(self.paths)


class TrainISTANet:
    def __init__(self, path_images: Iterable, crop_size: int, ds_factor: int, Qinit: torch.Tensor = None):
        """Defines the training procedure.

        Args:
            `path_images` (Iterable): Each element should contain the path to a single training image.
            `crop_size` (int): The size of each patch extracted from the training image.
            `ds_factor` (int): The downsampling factor. Must be divisible by `crop_size`
            `Qinit` (torch.Tensor, optional): _description_. Defaults to None.
        """
        self.path_images = path_images
        self.crop_size = crop_size
        self.dataset = ImageData(path_images, crop_size=crop_size)

        phi = getDSMatrix(res_init=crop_size, ds_factor=ds_factor)
        self.phi = torch.tensor(phi).float()

        if type(Qinit) != torch.Tensor:
            print("No valid Qinit provided. Calculating...")
            self._getQinit()
        else:
            print("Valid Qinit found!")
            self._setQinit(Qinit)

    def initModel(self, ista_layers: int, filter_count: int, filter_size: int, cha_color: int = 3):
        """Initilises the ISTA-Net model.

        Args:
            `ista_layers` (int): Number of "ISTA-Modules".
            `filter_count` (int): Maximum number of filters.
            `filter_size` (int): Size of each filter.
            `cha_color` (int, optional): Number of color channels.
        """
        self.model = ISTANet(ista_layers=ista_layers, filter_count=filter_count, filter_size=filter_size, cha_color=cha_color)

    def _setQinit(self, Qinit: torch.Tensor):
        if type(Qinit) != torch.Tensor:
            raise Exception(f"Qinit should be Tensor! ({type(Qinit)=} != torch.Tensor)")
        self.Qinit = Qinit

    def _getQinit(self):
        rng = np.random.default_rng()
        X = torch.zeros(size=(self.crop_size**2, len(self.path_images)))
        pre_process = Compose([ToTensor(), RandomCrop(size=(self.crop_size, self.crop_size))])
        for i in trange(X.shape[1]):
            path_img = self.path_images[rng.integers(low=0, high=len(self.path_images))]
            with Image.open(path_img) as img:
                img = pre_process(img).float()
                X[:, i] = img[rng.integers(0, 2, endpoint=True)].flatten()  # pick r, g, or b

        Y = self.phi @ X
        self.Qinit = X @ Y.T @ torch.linalg.inv(Y @ Y.T)

    def toDevice(self, device: str = "cpu"):
        """Allows the training to be performed on the CPU or a single GPU.

        Args:
            `device` (str, optional): Device which should run the model. Defaults to "cpu".
        """
        self.device = device
        self.Qinit = self.Qinit.to(device)
        self.phi = self.phi.to(device)
        self.model = self.model.to(device)

    def start(self, epochs: int, learning_rate: float, batch_size: int, weight_discrepency: float = 1.0, weight_constraint: float = 0.01):
        """Start the training

        Args:
            `epochs` (int): Number of epochs to run.
            `learning_rate` (float): Learning rate.
            `batch_size` (int): Size of the minibatches.
            `weight_discrepency` (float, optional): How much to weight the MSE loss (L_D). Defaults to 1.0.
            `weight_constraint` (float, optional): How much to weight the inverse morphism loss (L_IM). Defaults to 0.01.
        """
        optimiser = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate)
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.A = self.phi.T @ self.phi

        pbar_epochs = tqdm(total=epochs, desc="Epoch")
        pbar_batch = tqdm(total=len(dataloader), leave=True, desc="Batch")

        for _ in range(epochs):
            loss_sum = 0
            for X_HR in dataloader:
                optimiser.zero_grad()

                X_HR = X_HR.to(self.device).flatten(-2)
                Y = linear(X_HR, self.phi, bias=None)
                B = linear(Y, self.phi.T, bias=None)

                X_SR, error_symmetry = self.model(Y, self.A, B, self.Qinit)

                loss_discrepancy = mse_loss(X_SR, X_HR)
                loss_constraint = torch.mean(error_symmetry[0] ** 2)
                for i in range(1, self.model.T):
                    loss_constraint += torch.mean((error_symmetry[i] ** 2))

                loss_batch = weight_discrepency * loss_discrepancy + 1 / self.model.T * weight_constraint * loss_constraint

                loss_sum += loss_batch.item()
                loss_batch.backward()
                optimiser.step()
                pbar_batch.set_postfix_str(f"L={loss_batch.item():.2e}|L_D={loss_discrepancy.item():.2e}|L_IM={loss_constraint.item():.2e}")
                pbar_batch.update()

            loss_mean = loss_sum / len(dataloader)
            pbar_epochs.set_postfix_str(f"Average Loss={loss_mean:.2e}")
            pbar_epochs.update()
            pbar_batch.reset()

        pbar_epochs.close()
        pbar_batch.close()
        torch.save(self.model.state_dict(), f"ISTANet_e{epochs}.pth")


class TrainISTA2vec:
    def __init__(self, path_images: Iterable, crop_size: int, ds_factor: int):
        """Defines the training procedure.

        Args:
            `path_images` (Iterable): Each element should contain the path to a single training image.
            `crop_size` (int): The size of each patch extracted from the training image.
            `ds_factor` (int): The downsampling factor. Must be divisible by `crop_size`
        """
        self.path_images = path_images
        self.crop_size = crop_size
        self.dataset = ImageData(path_images, crop_size=crop_size)
        self.rng = np.random.default_rng()

        phi = getDSMatrix(res_init=crop_size, ds_factor=ds_factor)
        self.phi = torch.tensor(phi).float()

    def initModel(self, ista_layers: int, filter_count: int, filter_size: int, cha_color: int = 3):
        """Initilises the ISTA-Net model.

        Args:
            `ista_layers` (int): Number of "ISTA-Modules".
            `filter_count` (int): Maximum number of filters.
            `filter_size` (int): Size of each filter.
            `cha_color` (int, optional): Number of color channels.
        """
        self.model_student = ISTANet(ista_layers=ista_layers, filter_count=filter_count, filter_size=filter_size, cha_color=cha_color)
        self.model_teacher = deepcopy(self.model_student)

    def toDevice(self, device: str = "cpu"):
        """Allows the training to be performed on the CPU or a single GPU.

        Args:
            `device` (str, optional): Device which should run the model. Defaults to "cpu".
        """
        self.device = device
        self.phi = self.phi.to(device)
        self.model_student = self.model_student.to(device)
        self.model_teacher = self.model_teacher.to(device)

    def start(self, epochs: int, learning_rate: float, batch_size: int, mask_size: int, tau: tuple, weight_constraint: float = 0.01):
        """Start the training

        Args:
            `epochs` (int): Number of epochs to run.
            `learning_rate` (float): Learning rate.
            `batch_size` (int): Size of the minibatches.
            `mask_size` (int): Size of the mask.
            `tau` (tuple): Annealing of weight sharing. Linearly anneals between tau[0] and tau[1].
            `weight_constraint` (float, optional): How much to weight the inverse morphism loss (L_IM). Defaults to 0.01.
        """
        optimiser = torch.optim.Adam(params=self.model_student.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.A = self.phi.T @ self.phi

        pbar_epochs = tqdm(total=epochs, desc="Epoch")
        pbar_batch = tqdm(total=len(dataloader), leave=True, desc="Batch")

        self.tau = torch.linspace(tau[0], tau[1], steps=epochs, requires_grad=False).to(device)
        for e in range(epochs):
            loss_sum = 0
            for X_HR in dataloader:
                optimiser.zero_grad()

                X_HR = X_HR.to(self.device).flatten(-2)
                Y = linear(X_HR, self.phi, bias=None)
                B = linear(Y, self.phi.T, bias=None)
                with torch.no_grad():
                    X_SR_T, _ = self.model_teacher(Y, self.A, B, self.Qinit)
                    mask = torch.zeros(size=(*X_SR_T.shape[:-1], self.crop_size, self.crop_size), dtype=torch.bool)
                    for m, idx in zip(mask, self.rng.integers(low=0, high=self.crop_size - mask_size, size=mask.shape[0], endpoint=True)):
                        m[..., idx : idx + mask_size, idx : idx + mask_size] = True
                    mask = mask.flatten(-2)

                X_SR_S, error_symmetry = self.model_student(Y, self.A, B, self.Qinit)

                loss_discrepancy = mse_loss(X_SR_T, X_SR_S)
                loss_constraint = torch.mean(error_symmetry[0] ** 2)
                for i in range(1, self.model_student.T):
                    loss_constraint += torch.mean((error_symmetry[i] ** 2))

                loss_batch = loss_discrepancy + 1 / self.model_student.T * weight_constraint * loss_constraint

                loss_sum += loss_batch.item()
                loss_batch.backward()
                optimiser.step()
                pbar_batch.set_postfix_str(f"L={loss_batch.item():.2e}|L_D={loss_discrepancy.item():.2e}|L_IM={loss_constraint.item():.2e}")
                pbar_batch.update()

            loss_mean = loss_sum / len(dataloader)
            pbar_epochs.set_postfix_str(f"Average Loss={loss_mean:.2e}")
            pbar_epochs.update()
            pbar_batch.reset()

            scheduler.step()

            with torch.no_grad():
                states_teacher = deepcopy(self.model_teacher.state_dict())
                states_student = deepcopy(self.model_student.state_dict())
                for parameter, weight_s in states_student.items():
                    weight_t = states_teacher[parameter]
                    weight_t = tau[e] * weight_t + (1 - tau[e]) * weight_s
                    states_teacher[parameter] = weight_t

                self.model_teacher.load_state_dict(states_teacher)

        pbar_epochs.close()
        pbar_batch.close()
        torch.save(self.model_student.state_dict(), f"ista2vec_e{epochs}.pth")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    files = []
    Qinit = None

    training = TrainISTANet(path_images=files, crop_size=32, ds_factor=2, Qinit=Qinit)
    training.initModel(ista_layers=9, filter_count=32, filter_size=3, cha_color=3)
    training.toDevice(device=device)
    training.start(epochs=500, learning_rate=1e-4, batch_size=64)
