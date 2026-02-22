# CellViT Training: Data Flow, Sample Format, and Evaluation

This report explains how the CellViT model is trained in this codebase: what data it receives, the structure of each batch, model outputs, and how results are evaluated.

---

## 1. Training Pipeline Overview

**Entry point:** `run_cellvit.py`

- Parses CLI via `ExperimentBaseParser` and selects the experiment class by dataset name (`pannuke` → `ExperimentCellVitPanNuke`, `conic` → `ExperimentCellViTCoNic`).
- Instantiates the experiment and calls `experiment.run_experiment()`.
- After training, runs inference with `InferenceCellViT` (optional).

**Experiment:** `cell_segmentation/experiments/experiment_cellvit_pannuke.py`

- Loads dataset config from `dataset_path/dataset_config.yaml` (e.g. `tissue_types`, `nuclei_types`).
- Builds loss dict, model, optimizer, scheduler, (optional) early stopping.
- Gets **transforms** via `get_transforms(transformations, input_shape)` (default `input_shape=256`).
- Gets **datasets** via `get_datasets(train_transforms, val_transforms)` which uses `select_dataset("pannuke", split, dataset_config, transforms)` → `PanNukeDataset`.
- Builds **DataLoaders** (train with optional `WeightedRandomSampler`, val with `batch_size=128`).
- Instantiates **CellViTTrainer** and calls `trainer.fit(epochs, train_dataloader, val_dataloader, ...)`.

**Trainer:** `cell_segmentation/trainer/trainer_cellvit.py`

- `fit()` alternates `train_epoch()` and `validation_epoch()`.
- Each **train_epoch** iterates `train_dataloader`, calls **train_step(batch, ...)** for each batch: forward → unpack_predictions / unpack_masks → calculate_loss → backward → **calculate_step_metric_train**.
- Each **validation_epoch** iterates `val_dataloader`, calls **validation_step** → **calculate_step_metric_validation** (includes PQ).

So: **model is trained** by supervised multi-task learning on **patch images** with **multiple mask/label branches**; **goodness** is measured by **loss** (per-branch) and **metrics** (Dice, Jaccard, tissue accuracy, and on validation **Panoptic Quality (PQ)**).

---

## 2. Data Source and Sample Structure (Dataset)

**Dataset class:** `cell_segmentation/datasets/pannuke.py` → `PanNukeDataset`

**Data layout on disk (PanNuke):**

- Per fold: `fold{N}/images/*.png`, `fold{N}/labels/<stem>.npy`, `fold{N}/types.csv` (img → tissue type string).
- Each `.npy` mask is a dict: `inst_map` (H, W) instance IDs, `type_map` (H, W) nuclei type per pixel. Loaded as stacked array `(H, W, 2)` = [inst_map, type_map].

**`__getitem__(index)` returns a tuple of 4 elements:**

| Index | Content | Type | Shape / Format |
|-------|---------|------|----------------|
| 0 | Image | `torch.Tensor` | `(3, H, W)` — default ** (3, 256, 256) ** after transforms |
| 1 | Masks | `dict` | See below |
| 2 | Tissue type | `str` | e.g. `"Breast"`, `"Liver"` |
| 3 | Image name | `str` | e.g. `"0_0.png"` |

**Image:**

- Loaded as RGB numpy `(H, W, 3)`, then transformed (Albumentations: e.g. RandomRotate90, flips, Normalize).
- Converted to tensor, permuted to `(3, H, W)`, normalized to [0,1] if max ≥ 5.
- **Typical shape per sample:** `(3, 256, 256)`.

**Masks dict (per sample):**

| Key | Shape | Description |
|-----|--------|-------------|
| `instance_map` | `(H, W)` int64 | Instance ID per pixel (0 = background, 1,2,… = cell instances) |
| `nuclei_type_map` | `(H, W)` int64 | Nuclei class per pixel (same as type_map; 0=Background, 1=Neoplastic, …) |
| `nuclei_binary_map` | `(H, W)` int64 | Binary mask: 0 or 1 (derived from instance_map > 0) |
| `hv_map` | `(2, H, W)` float32 | Horizontal/vertical gradient maps for instance separation (from `gen_instance_hv_map`) |

Optional (if `stardist=True` / `regression=True` in config): `dist_map`, `stardist_map`, `regression_map` (see docstring in `pannuke.py`).

So a **single sample** is:

- **Input:** image `(3, 256, 256)`, masks dict (all 256×256 or 2×256×256), tissue string, filename.
- **Spatial size** is determined by transforms; default config uses **256×256** patches.

---

## 3. What the DataLoader Reads (Batch Structure)

DataLoader uses PyTorch’s **default collate**: tensors are stacked along a new batch dimension; dicts are collated by stacking each tensor value; lists of strings stay as lists.

So each **batch** is a tuple of 4 elements:

| Batch index | Content | Shape / Format |
|-------------|---------|----------------|
| **batch[0]** | Images | `(B, 3, H, W)` — e.g. ** (B, 3, 256, 256) ** |
| **batch[1]** | Masks dict | Each value is stacked: |
| | `instance_map` | `(B, H, W)` |
| | `nuclei_type_map` | `(B, H, W)` |
| | `nuclei_binary_map` | `(B, H, W)` |
| | `hv_map` | `(B, 2, H, W)` |
| **batch[2]** | Tissue types | List of B strings, e.g. `["Breast", "Liver", ...]` |
| **batch[3]** | Image names | List of B strings |

**Typical values:** `B = 32` (train, from config `batch_size`), `B = 128` (validation); `H = W = 256` with default `input_shape`.

Code reference (trainer): `trainer_cellvit.py`:

```python
imgs = batch[0].to(self.device)       # (B, 3, H, W)
masks = batch[1]                       # dict of (B, ...) tensors
tissue_types = batch[2]                # list of str, length B
```

---

## 4. Model Input and Output

**Model:** e.g. `CellViT` in `models/segmentation/cell_segmentation/cellvit.py`.

**Input:**

- `x`: `(B, 3, H, W)`; H, W must be divisible by `patch_size` (e.g. 16 for 256).

**Forward output (raw) — dict of tensors:**

| Key | Shape | Description |
|-----|--------|-------------|
| `tissue_types` | `(B, num_tissue_classes)` | Logits for tissue classification (e.g. 19 for PanNuke) |
| `nuclei_binary_map` | `(B, 2, H, W)` | Logits for binary nuclei (background vs nuclei) |
| `hv_map` | `(B, 2, H, W)` | HV map logits (horizontal / vertical) |
| `nuclei_type_map` | `(B, num_nuclei_classes, H, W)` | Logits for nuclei types (e.g. 6 classes including background) |
| `regression_map` | `(B, 2, H, W)` | Optional; only if `regression_loss=True` |

So the **model output** is multi-branch: one classification head (tissue) and three segmentation-style heads (binary map, hv map, nuclei type map), plus optional regression map.

---

## 5. Post-processing of Predictions and Ground Truth

**Unpack predictions** (`unpack_predictions` in trainer):

- `nuclei_binary_map`: softmax over channel dim → `(B, 2, H, W)`.
- `nuclei_type_map`: softmax → `(B, num_nuclei_classes, H, W)`.
- **Instance map** is computed from predictions: `model.calculate_instance_map(predictions, magnification)` → instance segmentation from hv + binary + type.
- `instance_types_nuclei`: from `model.generate_instance_nuclei_map(instance_map, instance_types)`.

**Unpack masks (GT)** (`unpack_masks`):

- `nuclei_binary_map` → one-hot `(B, 2, H, W)` (background, nuclei).
- `nuclei_type_map` → one-hot `(B, num_nuclei_classes, H, W)`.
- `hv_map`, `instance_map` moved to device; tissue strings converted to class indices via `dataset_config["tissue_types"]` → `(B,)` LongTensor.

Both predictions and GT are then wrapped in a **DataclassHVStorage** for a unified interface in loss and metrics.

---

## 6. How Results Are Judged (Loss and Metrics)

### 6.1 Loss (training and validation)

**Loss is per-branch, then summed.**

Branches and typical loss functions (from experiment `get_loss_fn`):

- **nuclei_binary_map:** e.g. BCE + Dice (xentropy_loss, dice_loss).
- **hv_map:** e.g. MSE + MSGE (mse_loss_maps, msge_loss_maps).
- **nuclei_type_map:** e.g. BCE + Dice.
- **tissue_types:** CrossEntropy.
- **regression_map** (optional): e.g. MSE.

Each term is weighted; total loss = sum of (weight × loss) over all branches. So **“good”** means low total loss and low per-branch loss.

### 6.2 Training-step metrics (`calculate_step_metric_train`)

- **Tissue:** argmax of tissue logits vs GT tissue index → **accuracy** (per batch).
- **Binary nuclei:** for each image in the batch, compare predicted binary map (argmax of 2-channel) to GT binary map:
  - **Dice** (ignoring background index 0).
  - **Binary Jaccard (IoU)**.
- Aggregated over epoch: mean Dice, mean Jaccard, tissue accuracy (and optionally logged per-batch).

### 6.3 Validation-step metrics (`calculate_step_metric_validation`)

Same as train plus:

- **Panoptic Quality (PQ):**
  - **bPQ:** PQ between predicted instance map and GT instance map (after `remap_label`), via `get_fast_pq`.
  - **mPQ:** mean of per–nuclei-class PQ (each class’s instance map vs GT), with `match_iou=0.5`; NaN when GT has no instances for that class.

So **“good”** at validation is: high Binary-Cell-Dice, high Binary-Cell-Jaccard, high tissue accuracy, high **bPQ** and **mPQ**. Early stopping / model selection can use e.g. validation PQ or loss (config-dependent).

---

## 7. Code References (Key Files and Symbols)

| Topic | File | Symbol / Location |
|-------|------|-------------------|
| Entry | `cell_segmentation/run_cellvit.py` | `main` → `ExperimentCellVitPanNuke`, `run_experiment()` |
| Experiment setup | `cell_segmentation/experiments/experiment_cellvit_pannuke.py` | `run_experiment()`, `get_datasets()`, `get_transforms()`, `get_loss_fn()`, `get_train_model()` |
| Dataset | `cell_segmentation/datasets/pannuke.py` | `PanNukeDataset.__getitem__()`, `load_imgfile()`, `load_maskfile()`, `gen_instance_hv_map()` |
| Dataset selection | `cell_segmentation/datasets/dataset_coordinator.py` | `select_dataset()` |
| Trainer loop | `cell_segmentation/trainer/trainer_cellvit.py` | `train_epoch()`, `train_step()`, `validation_epoch()`, `validation_step()` |
| Batch unpacking | `cell_segmentation/trainer/trainer_cellvit.py` | `train_step()`: `batch[0]`, `batch[1]`, `batch[2]` |
| Predictions / GT | `cell_segmentation/trainer/trainer_cellvit.py` | `unpack_predictions()`, `unpack_masks()` |
| Loss | `cell_segmentation/trainer/trainer_cellvit.py` | `calculate_loss()` |
| Train metrics | `cell_segmentation/trainer/trainer_cellvit.py` | `calculate_step_metric_train()` |
| Val metrics (incl. PQ) | `cell_segmentation/trainer/trainer_cellvit.py` | `calculate_step_metric_validation()`, `get_fast_pq`, `remap_label` |
| Model forward | `models/segmentation/cell_segmentation/cellvit.py` | `CellViT.forward()` |
| Dataset config | `configs/datasets/PanNuke/dataset_config.yaml` | `tissue_types`, `nuclei_types` |
| Train config | `configs/examples/cell_segmentation/train_cellvit.yaml` | `data.input_shape`, `data.num_nuclei_classes`, `training.batch_size`, `loss` |

---

## 8. Summary Table

| Question | Answer |
|----------|--------|
| **How is the model trained?** | Supervised multi-task: one forward pass per batch; loss = sum of weighted losses over nuclei_binary_map, hv_map, nuclei_type_map, tissue_types (and optionally regression_map); backward and optimizer step. |
| **What does one sample look like?** | Image `(3, 256, 256)`, dict of masks (instance, nuclei type, binary, hv_map all 256×256 or 2×256×256), tissue type string, filename. |
| **What does one batch look like?** | `batch[0]` = `(B, 3, 256, 256)`, `batch[1]` = dict of `(B, ...)` tensors, `batch[2]` = list of B tissue strings, `batch[3]` = list of B filenames. |
| **What is the model output?** | Dict: tissue logits `(B, num_tissue_classes)`, nuclei_binary logits `(B, 2, H, W)`, hv logits `(B, 2, H, W)`, nuclei_type logits `(B, num_nuclei_classes, H, W)`, plus optional regression_map. After unpack: instance map and instance-type maps. |
| **How is “good” judged?** | **Loss:** lower is better (per-branch and total). **Metrics:** higher is better — Binary Dice, Binary Jaccard, tissue accuracy; on validation also **bPQ** and **mPQ** (Panoptic Quality). |

This should give a complete picture of data format, dimensions, and evaluation for CellViT training in this repository.
