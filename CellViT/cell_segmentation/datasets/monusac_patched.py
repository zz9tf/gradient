# -*- coding: utf-8 -*-
# MoNuSAC Patched Dataset
#
# Reads hover_net-style .npy patches (256,256,5): RGB + inst_map + type_map.
# Aligns with hover_net extract_patches.py output for shared data pipeline.
# Dataset layout: dataset_path / {train|valid|test} / patch_subdir / *.npy
#
# @ See docs/readmes and plan: MoNuSAC CellViT alignment

import logging
from pathlib import Path
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from natsort import natsorted

from cell_segmentation.datasets.base_cell import CellDataset
from cell_segmentation.datasets.pannuke import PanNukeDataset

logger = logging.getLogger()
logger.addHandler(logging.NullHandler())

# Default split subdir names (hover_net extract_patches: train, valid, test)
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_VAL_SPLIT = "valid"
DEFAULT_TEST_SPLIT = "test"
DEFAULT_PATCH_SUBDIR = "256x256_164x164"
DEFAULT_TISSUE_TYPE = "MoNuSAC"


class MoNuSACPatchedDataset(CellDataset):
    """Dataset that reads hover_net-style .npy patches for CellViT training.

    Each .npy has shape (H, W, 5): [..., :3] RGB, [..., 3] inst_map, [..., 4] type_map.
    Produces the same (image, masks_dict, tissue_type, name) interface as PanNukeDataset.
    """

    def __init__(
        self,
        dataset_path: Union[Path, str],
        split: str,
        transforms: Callable = None,
        stardist: bool = False,
        regression: bool = False,
        train_split: str = DEFAULT_TRAIN_SPLIT,
        val_split: str = DEFAULT_VAL_SPLIT,
        test_split: str = DEFAULT_TEST_SPLIT,
        patch_subdir: str = DEFAULT_PATCH_SUBDIR,
        tissue_type: str = DEFAULT_TISSUE_TYPE,
    ) -> None:
        """Build dataset from dataset_path / split_subdir / patch_subdir / *.npy.

        Args:
            dataset_path: Root path (e.g. .../dataset/monusac).
            split: One of "train", "validation", "test".
            transforms: Albumentations-style (image=..., mask=...) transform.
            stardist: If True, add dist_map and stardist_map to masks.
            regression: If True, add regression_map to masks.
            train_split: Subdir name for training (e.g. "train").
            val_split: Subdir name for validation (e.g. "valid").
            test_split: Subdir name for test (e.g. "test").
            patch_subdir: Subdir under each split (e.g. "256x256_164x164").
            tissue_type: String returned as tissue type for all samples.
        """
        self.dataset_path = Path(dataset_path).resolve()
        self.transforms = transforms
        self.stardist = stardist
        self.regression = regression
        self.tissue_type = tissue_type
        self.cell_count = None

        split_subdir = {
            "train": train_split,
            "validation": val_split,
            "val": val_split,
            "test": test_split,
        }.get(split.lower(), train_split)
        self.patch_dir = self.dataset_path / split_subdir / patch_subdir
        if not self.patch_dir.is_dir():
            raise FileNotFoundError(
                f"MoNuSAC patch directory not found: {self.patch_dir}"
            )
        self.npy_files: List[Path] = [
            f for f in natsorted(self.patch_dir.glob("*.npy")) if f.is_file()
        ]
        logger.info(
            "Created MoNuSACPatchedDataset split=%s path=%s len=%s",
            split,
            str(self.patch_dir),
            len(self.npy_files),
        )

    def __len__(self) -> int:
        return len(self.npy_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict, str, str]:
        """Return (image_tensor, masks_dict, tissue_type, filename)."""
        path = self.npy_files[index]
        data = np.load(path, allow_pickle=False)
        if data.shape != (256, 256, 5):
            raise ValueError(
                f"Expected npy shape (256,256,5), got {data.shape} at {path}"
            )
        img = data[..., :3].astype(np.uint8)
        inst_map = data[..., 3].astype(np.int32)
        type_map = data[..., 4].astype(np.int32)
        mask = np.stack([inst_map, type_map], axis=-1)

        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        inst_map = mask[:, :, 0].copy()
        type_map = mask[:, :, 1].copy()
        np_map = mask[:, :, 0].copy()
        np_map[np_map > 0] = 1
        hv_map = PanNukeDataset.gen_instance_hv_map(inst_map)

        img_t = torch.Tensor(img).type(torch.float32)
        img_t = img_t.permute(2, 0, 1)
        if torch.max(img_t) >= 5:
            img_t = img_t / 255.0

        masks = {
            "instance_map": torch.Tensor(inst_map).type(torch.int64),
            "nuclei_type_map": torch.Tensor(type_map).type(torch.int64),
            "nuclei_binary_map": torch.Tensor(np_map).type(torch.int64),
            "hv_map": torch.Tensor(hv_map).type(torch.float32),
        }
        if self.stardist:
            dist_map = PanNukeDataset.gen_distance_prob_maps(inst_map)
            stardist_map = PanNukeDataset.gen_stardist_maps(inst_map)
            masks["dist_map"] = torch.Tensor(dist_map).type(torch.float32)
            masks["stardist_map"] = torch.Tensor(stardist_map).type(torch.float32)
        if self.regression:
            masks["regression_map"] = PanNukeDataset.gen_regression_map(inst_map)

        return img_t, masks, self.tissue_type, path.name

    def set_transforms(self, transforms: Callable) -> None:
        self.transforms = transforms

    def load_cell_count(self) -> None:
        """No cell_count CSV for patched npy; set placeholder for weighted sampling."""
        self.cell_count = None

    def get_sampling_weights_tissue(self, gamma: float = 1) -> torch.Tensor:
        """Uniform weights (single tissue type)."""
        return torch.ones(self.__len__(), dtype=torch.float32)

    def get_sampling_weights_cell(self, gamma: float = 1) -> torch.Tensor:
        """Uniform weights when cell_count not available."""
        return torch.ones(self.__len__(), dtype=torch.float32)
