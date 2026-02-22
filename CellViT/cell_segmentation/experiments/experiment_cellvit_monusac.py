# -*- coding: utf-8 -*-
# CellViT Experiment for MoNuSAC (patched npy from hover_net pipeline)
#
# Uses MoNuSACPatchedDataset; data layout: dataset_path / {train,valid,test} / patch_subdir / *.npy

import copy
from pathlib import Path
from typing import Callable, Tuple, Union

import torch
from torch.utils.data import Dataset

from cell_segmentation.datasets.dataset_coordinator import select_dataset
from cell_segmentation.experiments.experiment_cellvit_pannuke import (
    ExperimentCellVitPanNuke,
)


class ExperimentCellVitMoNuSAC(ExperimentCellVitPanNuke):
    """CellViT experiment for MoNuSAC patched .npy data (same source as hover_net)."""

    def get_datasets(
        self,
        train_transforms: Callable = None,
        val_transforms: Callable = None,
    ) -> Tuple[Dataset, Dataset]:
        """Return train and validation datasets from train_split and val_split subdirs."""
        if "regression_loss" in self.run_conf["model"].keys():
            self.run_conf["data"]["regression_loss"] = True

        train_dataset = select_dataset(
            dataset_name="monusac_patched",
            split="train",
            dataset_config=self.run_conf["data"],
            transforms=train_transforms,
        )
        val_dataset = select_dataset(
            dataset_name="monusac_patched",
            split="validation",
            dataset_config=self.run_conf["data"],
            transforms=val_transforms,
        )
        return train_dataset, val_dataset
