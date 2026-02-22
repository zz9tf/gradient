# Config Files in cell_segmentation

## Difference Between the Three Configs

| Config | Purpose | Main difference |
|--------|--------|-----------------|
| **train_cellvit.yaml** | Single CellViT (HoverNet-style) training run | Full template for one run: model (ViT/SAM), loss (nuclei_binary_map, hv_map, nuclei_type_map, tissue_types), training, logging, transformations. Uses `data.train_folds` / `val_folds` or `val_split` (PanNuke/Conic style). |
| **train_cellvit_sweep.yaml** | Hyperparameter sweep (WandB) for CellViT | Same structure as `train_cellvit.yaml` plus a **sweep** section and **parameters** under e.g. `training.parameters` and `optimizer_hyperparameter.parameters` to define search space (e.g. lr min/max, drop_rate values). Used with `--sweep` or `--agent`. |
| **train_stardist.yaml** | Single StarDist-style model training | Different **model** (e.g. backbone RN50, `n_rays`) and **loss** branches (`dist_map`, `stardist_map`). Scheduler can use `reducelronplateau`. No `extract_layers`; otherwise similar to train_cellvit. |

## Data Section: PanNuke/Conic vs MoNuSAC (monusac_patched)

- **PanNuke/Conic**: use `train_folds`, `val_folds`, `test_folds` and/or `val_split` (float). No `train_split`/`val_split` as folder names.
- **MoNuSAC (monusac_patched)**: use `train_split` and `val_split` as **subdir names** (e.g. `train`, `valid`) and `patch_subdir` (e.g. `256x256_164x164`). Path layout: `dataset_path / {train,valid,test} / patch_subdir / *.npy`.

Your `train_config.yaml` is for **monusac_patched** and already has the correct data fields; it only needs the rest of the run (logging, gpu, model, loss, training, transformations) to start a single CellViT run.
