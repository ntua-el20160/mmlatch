import os

from typing import Optional, Any, cast, Callable

import torch

import numpy as np
import torch
import validators
import yaml
import pickle

from torch.optim.optimizer import Optimizer

from typing import Dict, Union, List, TypeVar, Tuple

import numpy as np
import matplotlib.pyplot as plt
import umap
import seaborn as sns
import re


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

NdTensor = Union[np.ndarray, torch.Tensor, List[T]]

Device = Union[torch.device, str]

ModuleOrOptimizer = Union[torch.nn.Module, Optimizer]

# word2idx, idx2word, embedding vectors
Embeddings = Tuple[Dict[str, int], Dict[int, str], np.ndarray]

ValidationResult = Union[validators.ValidationFailure, bool]

GenericDict = Dict[K, V]


def is_file(inp: Optional[str]) -> ValidationResult:
    if not inp:
        return False
    return os.path.isfile(inp)


def to_device(
    tt: torch.Tensor, device: Optional[Device] = "cpu", non_blocking: bool = False
) -> torch.Tensor:
    return tt.to(device, non_blocking=non_blocking)


def t_(
    data: NdTensor,
    dtype: torch.dtype = torch.float,
    device: Optional[Device] = "cpu",
    requires_grad: bool = False,
) -> torch.Tensor:
    """Convert a list or numpy array to torch tensor. If a torch tensor
    is passed it is cast to  dtype, device and the requires_grad flag is
    set IN PLACE.

    Args:
        data: (list, np.ndarray, torch.Tensor): Data to be converted to
            torch tensor.
        dtype: (torch.dtype): The type of the tensor elements
            (Default value = torch.float)
        device: (torch.device, str): Device where the tensor should be
            (Default value = 'cpu')
        requires_grad: bool): Trainable tensor or not? (Default value = False)

    Returns:
        (torch.Tensor): A tensor of appropriate dtype, device and
            requires_grad containing data

    """

    if isinstance(device, str):
        device = torch.device(device)

    tt = torch.as_tensor(data, dtype=dtype, device=device).requires_grad_(requires_grad)

    return tt


def t(
    data: NdTensor,
    dtype: torch.dtype = torch.float,
    device: Device = "cpu",
    requires_grad: bool = False,
) -> torch.Tensor:
    """Convert a list or numpy array to torch tensor. If a torch tensor
    is passed it is cast to  dtype, device and the requires_grad flag is
    set. This always copies data.

    Args:
        data: (list, np.ndarray, torch.Tensor): Data to be converted to
            torch tensor.
        dtype: (torch.dtype): The type of the tensor elements
            (Default value = torch.float)
        device: (torch.device, str): Device where the tensor should be
            (Default value = 'cpu')
        requires_grad: (bool): Trainable tensor or not? (Default value = False)

    Returns:
        (torch.Tensor): A tensor of appropriate dtype, device and
            requires_grad containing data

    """
    tt = torch.tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

    return tt


def mktensor(
    data: NdTensor,
    dtype: torch.dtype = torch.float,
    device: Device = "cpu",
    requires_grad: bool = False,
    copy: bool = True,
) -> torch.Tensor:
    """Convert a list or numpy array to torch tensor. If a torch tensor
        is passed it is cast to  dtype, device and the requires_grad flag is
        set. This can copy data or make the operation in place.

    Args:
        data: (list, np.ndarray, torch.Tensor): Data to be converted to
            torch tensor.
        dtype: (torch.dtype): The type of the tensor elements
            (Default value = torch.float)
        device: (torch.device, str): Device where the tensor should be
            (Default value = 'cpu')
        requires_grad: (bool): Trainable tensor or not? (Default value = False)
        copy: (bool): If false creates the tensor inplace else makes a copy
            (Default value = True)

    Returns:
        (torch.Tensor): A tensor of appropriate dtype, device and
            requires_grad containing data

    """
    tensor_factory = t if copy else t_

    return tensor_factory(data, dtype=dtype, device=device, requires_grad=requires_grad)


def from_checkpoint(
    checkpoint_file: Optional[str],
    obj: ModuleOrOptimizer,
    map_location: Optional[Device] = None,
) -> ModuleOrOptimizer:  # noqa: E501
    if checkpoint_file is None:
        return obj

    if not is_file(checkpoint_file):
        print(
            f"The checkpoint {checkpoint_file} you are trying to load "
            "does not exist. Continuing without loading..."
        )

        return obj
    state_dict = torch.load(checkpoint_file, map_location=map_location)

    if isinstance(obj, torch.nn.Module):
        if "model" in state_dict:
            state_dict = state_dict["model"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    if isinstance(obj, torch.optim.Optimizer) and "optimizer" in state_dict:
        state_dict = state_dict["optimizer"]
    obj.load_state_dict(state_dict)  # type: ignore

    return obj


def rotate_tensor(l: torch.Tensor, n: int = 1) -> torch.Tensor:
    return torch.cat((l[n:], l[:n]))


def shift_tensor(l: torch.Tensor, n: int = 1) -> torch.Tensor:
    out = rotate_tensor(l, n=n)
    out[-n:] = 0

    return out


def safe_mkdirs(path: str) -> None:
    """! Makes recursively all the directory in input path"""
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            print(e)
            raise IOError((f"Failed to create recursive directories: {path}"))


def yaml_load(fname: str) -> GenericDict:
    with open(fname, "r") as fd:
        data = yaml.load(fd, Loader=yaml.Loader) # Changed from data = yaml.load(fd)
    return data


def yaml_dump(data: GenericDict, fname: str) -> None:
    with open(fname, "w") as fd:
        yaml.dump(data, fd)


def pickle_load(fname: str) -> Any:
    with open(fname, "rb") as fd:
        data = pickle.load(fd)
    return data


def pickle_dump(data: Any, fname: str) -> None:
    with open(fname, "wb") as fd:
        pickle.dump(data, fd)


def pad_mask(lengths: torch.Tensor, max_length: Optional[int] = None, device="cpu"):
    """lengths is a torch tensor"""
    if max_length is None:
        max_length = cast(int, torch.max(lengths).item())
    max_length = cast(int, max_length)
    idx = torch.arange(0, max_length).unsqueeze(0).to(device)
    mask = (idx < lengths.unsqueeze(1)).float()
    return mask


def print_separator(
    symbol: str = "*", n: int = 10, print_fn: Callable[[str], None] = print
):
    print_fn(symbol * n)



# ==== Function to bin predictions ====
def bin_predictions(predictions):
    if isinstance(predictions, torch.Tensor):  # Check if input is a tensor
        predictions = predictions.detach().cpu().numpy()  # Move to CPU & convert to NumPy

    bins = [-np.inf, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, np.inf]
    labels = [-3, -2, -1, 0, 1, 2, 3]

    return np.digitize(predictions, bins, right=True) - 1  # Adjust index

# ==== Function to plot UMAP embeddings ====
def plot_umap(embeddings, predictions, title_before, title_after, ax1, ax2, save_path):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)

    # Fit and transform embeddings
    embedding_before = reducer.fit_transform(embeddings["before"])
    embedding_after = reducer.fit_transform(embeddings["after"])

    # Scatter plots
    cmap = sns.color_palette("husl", 7)  # 7 color bins
    scatter1 = ax1.scatter(embedding_before[:, 0], embedding_before[:, 1], c=predictions, cmap="viridis", alpha=0.7)
    scatter2 = ax2.scatter(embedding_after[:, 0], embedding_after[:, 1], c=predictions, cmap="viridis", alpha=0.7)

    # Titles and aesthetics
    ax1.set_title(title_before)
    ax2.set_title(title_after)
    ax1.set_xticks([]), ax1.set_yticks([])
    ax2.set_xticks([]), ax2.set_yticks([])

    # Add colorbars
    plt.colorbar(scatter1, ax=ax1, label="Binned Prediction Labels")
    plt.colorbar(scatter2, ax=ax2, label="Binned Prediction Labels")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved to {save_path}")