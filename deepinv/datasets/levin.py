from __future__ import annotations
from os.path import join, basename
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy.io import loadmat
from torchvision.datasets.utils import download_and_extract_archive

from deepinv.datasets.base import ImageDataset
from .utils import resolve_root


class Levin2009(ImageDataset):
    """Dataset for `Understanding and evaluating blind deconvolution algorithms <https://doi.org/10.1109/CVPR.2009.5206815>`_

    Published in :footcite:t:`levin2009understanding`.

    The dataset consists of 4 sharp grayscale images (255x255) and 8 measured
    uniform blur kernels (support 10 to 25 pixels), giving 32 blurry images
    acquired with a real camera on a tripod. Each item returns the sharp image
    ``x``, the blurry image ``y`` and the ground truth kernel as the physics
    parameter ``filter``, compatible with :class:`deepinv.physics.Blur`.

    The dataset has no preferred ordering. The parameter ``ordering`` controls
    whether items are ordered by image first ``"image_first"`` or by kernel
    first ``"kernel_first"``. The method ``get_item`` selects an item directly
    by image index (1 to 4) and kernel index (1 to 8).

    :param Union[str, pathlib.Path] root: Root directory of the dataset.
    :param str ordering: ``"image_first"`` or ``"kernel_first"``.
    :param Callable transform: (optional) A function used to transform both the sharp and the blurry images. It is not applied to the kernels.
    :param bool download: Download the dataset.
    :param bool use_dict_output: whether to return output as dict with keys "x", "y", "params" instead of tuple (default `False`).

    |sep|

    :Examples:

        Download the dataset and load one of its elements ::

            from torchvision.transforms import ToTensor
            from deepinv.datasets import Levin2009
            dataset = Levin2009(root="datasets/Levin2009",
                                transform=ToTensor(),
                                download=True,
                                use_dict_output=True)
            sample = dataset[0]
            print(sample["x"].shape, sample["y"].shape, sample["params"]["filter"].shape)
    """

    _archive_url = (
        "https://www.wisdom.weizmann.ac.il/~levina/papers/LevinEtalCVPR09Data.zip"
    )
    _archive_md5 = "aba10d03ccdd9e01e9aa49744f3e552e"
    _folder = "Levin09blurdata"

    # The files are numbered im05..im08 and flit01..flit08.
    _image_ids = (5, 6, 7, 8)
    _kernel_ids = tuple(range(1, 9))

    def __init__(
        self,
        root: str | Path = None,
        ordering: str = "image_first",
        transform: Callable = None,
        download: bool = False,
        use_dict_output: bool = False,
    ) -> None:
        super().__init__(use_dict_output=use_dict_output)
        self.root = resolve_root(root, "Levin2009")
        self.ordering = ordering
        self.transform = transform

        if download:
            self.download(self.root)

    @classmethod
    def download(cls, root: str | Path = None, remove_finished: bool = False) -> None:
        """Download the dataset.

        :param Union[str, pathlib.Path] root: Root directory of the dataset.
        :param bool remove_finished: Remove the archive after extraction.
        """
        root = resolve_root(root, "Levin2009")
        download_and_extract_archive(
            cls._archive_url,
            root,
            filename=basename(cls._archive_url),
            md5=cls._archive_md5,
            remove_finished=remove_finished,
        )

    def __len__(self) -> int:
        return len(self._image_ids) * len(self._kernel_ids)

    def __getitem__(self, index: int):
        """Get a sharp image, a blurry image and the blur kernel.

        :param int index: Index of the item.

        :return: (torch.Tensor, torch.Tensor, dict) The sharp image, the blurry image and the dict ``{"filter": kernel}``. A dict ``{"x", "y", "params"}`` if ``use_dict_output``.
        """
        n_kernels = len(self._kernel_ids)
        if self.ordering == "image_first":
            image_index = index // n_kernels + 1
            kernel_index = index % n_kernels + 1
        elif self.ordering == "kernel_first":
            image_index = index % len(self._image_ids) + 1
            kernel_index = index // len(self._image_ids) + 1
        else:
            raise ValueError(f"Unsupported ordering: {self.ordering}")

        x, y, params = self.get_item(image_index, kernel_index)

        if self.use_dict_output:
            return {"x": x, "y": y, "params": params}
        return x, y, params

    def get_item(self, image_index: int, kernel_index: int):
        """Get an item by image index (1 to 4) and kernel index (1 to 8).

        :param int image_index: Index of the sharp image.
        :param int kernel_index: Index of the blur kernel.

        :return: (torch.Tensor, torch.Tensor, dict) The sharp image, the blurry image and the dict ``{"filter": kernel}``.
        """
        path = join(
            self.root,
            self._folder,
            f"im{self._image_ids[image_index - 1]:02d}_flit{kernel_index:02d}.mat",
        )
        data = loadmat(path)
        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.float32)
        f = data["f"].astype(np.float32)

        if self.transform is not None:
            x, y = self.transform(x), self.transform(y)

        k = torch.from_numpy(f)
        k = (k / k.sum())[None, None]  # (1, 1, h, w), sums to 1
        return x, y, {"filter": k}
