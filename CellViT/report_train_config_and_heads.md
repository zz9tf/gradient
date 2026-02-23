# CellViT 训练配置与模型结构说明报告

## 1. 配置与启动流程概览

### 1.1 配置文件 `train_config.yaml` 在做什么

本配置用于在 **MoNuSAC 切块数据（monusac_patched）** 上训练 **CellViT** 细胞分割网络，主要作用包括：

| 配置块 | 作用 |
|--------|------|
| **logging** | 离线模式、项目名 `cellvit-monusac`、日志目录、标签等 |
| **random_seed / gpu** | 固定随机种子 19，使用 GPU 0 |
| **data** | 数据集 `monusac_patched`，路径、train/valid 划分、patch 子目录 `256x256_164x164`，5 类细胞 + 1 类组织，输入尺寸 256 |
| **model** | 默认 backbone、无预训练、embed_dim=384、depth=12、6 heads、从第 3/6/9/12 层取特征、非共享 decoder、无回归损失 |
| **loss** | 为 4 个 head 配置损失：nuclei_binary_map（BCE+Dice）、hv_map（MSE+MSGE）、nuclei_type_map（BCE+Dice）、tissue_types（CE） |
| **training** | batch=4、100 epochs、AdamW、cosine 调度、随机采样、每 1 epoch 验证 |
| **transformations** | 随机 90° 旋转、水平/垂直翻转、归一化 mean/std=0.5 |
| **eval_checkpoint** | 评估时加载 `best_checkpoint.pth` |

### 1.2 启动入口 `run_cellvit.py`

- **入口**：`cell_segmentation/run_cellvit.py`，通过 `ExperimentBaseParser` 解析 YAML 与命令行，得到 `configuration`。
- **根据 `data.dataset` 选择实验类**：
  - `monusac_patched` → `ExperimentCellVitMoNuSAC`（继承自 `ExperimentCellVitPanNuke`）
- **流程**：创建对应 experiment → `run_experiment()`（数据、模型、trainer、训练）→ 训练结束后用 `InferenceCellViT` 做 patch 推理（`setup_patch_inference` + `run_patch_inference`）。

因此：**当前 config 的启动项就是 `run_cellvit.py`，且会走 MoNuSAC 实验类与标准 CellViT（HV 式）训练与推理流程。**

---

## 2. Input（模型输入）

- **来源**：DataLoader 的 batch 中 `batch[0]`，即 **图像张量**。
- **形状**：`(B, 3, H, W)`，例如 `(4, 3, 256, 256)`（B=batch_size，config 中为 4）。
- **数值范围**：经 dataset 与 `normalize` 变换后，一般为归一化后的 float（如 mean=0.5, std=0.5）。
- **数据管线**：
  - 数据集：`MoNuSACPatchedDataset`（`cell_segmentation/datasets/monusac_patched.py`）。
  - 每个样本从 `.npy` 读取 `(256,256,5)`：前 3 通道 RGB，第 4 通道 `inst_map`，第 5 通道 `type_map`；返回 `(image_tensor, masks_dict, tissue_type, filename)`，image 在 dataset 内已转为 `(3, H, W)` 的 float tensor。

**结论**：**Input = 归一化后的 RGB 图像，形状 (B, 3, 256, 256)。**

---

## 3. Output（模型输出）

模型 `forward` 返回一个 **字典**，键即各 head 的输出（及可选 token/regression）：

| 键 | 含义 | 形状（示例） |
|----|------|----------------|
| **tissue_types** | 组织类型 logits（来自 encoder 的 class token） | (B, num_tissue_classes)，当前 config 为 (B, 1) |
| **nuclei_binary_map** | 二类细胞分割 logits（前景/背景） | (B, 2, H, W) |
| **hv_map** | 水平-垂直图，用于实例分离 | (B, 2, H, W) |
| **nuclei_type_map** | 多类细胞类型 logits | (B, num_nuclei_classes, H, W)，当前为 (B, 5, H, W) |
| **regression_map** | 仅当 `regression_loss=True` 时存在 | (B, 2, H, W) |
| **tokens** | 仅当 `retrieve_tokens=True` 时存在 | 由 encoder 提供的 token 特征 |

训练时，这些原始输出会经 `unpack_predictions` 做 softmax、实例图计算等，再与 `unpack_masks` 得到的 GT 一起送入 `calculate_loss`。

**结论**：**Output = 字典，包含 tissue_types、nuclei_binary_map、hv_map、nuclei_type_map，以及可选的 regression_map / tokens。**

---

## 4. 有哪些 Head、在哪些文件里

CellViT（HV 版）有 **4 个任务头**，外加可选回归头。每个头对应一个“分支”：要么是 encoder 的 class token 分类，要么是 decoder 上的一支上采样 + 最后一层 1×1 卷积。

| Head 名称 | 类型 | 输出通道/含义 | 定义位置（文件: 位置说明） |
|-----------|------|----------------|----------------------------|
| **tissue_types** | 全局分类头 | num_tissue_classes（1） | **cellvit.py**：encoder 的 `ViTCellViT` 内部 class token → 线性分类；若用 `CellViTSAM` 等，则还有 `classifier_head` 在 **cellvit.py** 约 568–572 行 |
| **nuclei_binary_map** | 分割头（decoder 分支） | 2（背景/细胞） | **cellvit.py**：`branches_output["nuclei_binary_map"]`、`nuclei_binary_map_decoder`（`create_upsampling_branch(2+offset)`），forward 中 `_forward_upsample(..., nuclei_binary_map_decoder)` |
| **hv_map** | 分割头（decoder 分支） | 2（H 方向、V 方向） | **cellvit.py**：`hv_map_decoder`，`_forward_upsample(..., hv_map_decoder)` |
| **nuclei_type_map** | 分割头（decoder 分支） | num_nuclei_classes（5） | **cellvit.py**：`nuclei_type_maps_decoder`，`_forward_upsample(..., nuclei_type_maps_decoder)` |
| **regression_map**（可选） | 与 binary 共用分支 | 2 | **cellvit.py**：当 `regression_loss=True` 时，从 `nuclei_binary_map_decoder` 的 4 通道输出中切出后 2 通道作为 regression_map |

**共享 decoder 版本（CellViTShared）**：  
- **cellvit_shared.py**：同一套 skip 与上采样，分支仅为最后一层不同：`nuclei_binary_map_decoder`、`hv_map_decoder`、`nuclei_type_maps_decoder` 各为一个 `nn.Conv2d`，输出通道同上；tissue 仍为 classifier_head。

**结论**：  
- **标准 CellViT（非 shared）**：4 个头 + 可选 regression 均在 **`models/segmentation/cell_segmentation/cellvit.py`**。  
- **Shared 版本**：在 **`models/segmentation/cell_segmentation/cellvit_shared.py`**，头结构更简单（每分支一个 Conv2d）。

---

## 5. 若想“增加 Head”，应放在哪里、要做哪些事

要增加一个新 head，需要同时改 **模型输出**、**训练时的 GT/损失** 和 **配置**，并在 **推理/后处理** 中如需要则使用新输出。下面按顺序说明。

### 5.1 在模型里增加新 head（核心文件：`cellvit.py` 或 `cellvit_shared.py`）

1. **确定 head 类型**
   - **全局分类**（类似 tissue_types）：在 encoder 后接一个线性层或小 MLP，输入 class token，输出新类别数。
   - **逐像素图**（类似 nuclei_binary_map / hv_map / nuclei_type_map）：新增一条 decoder 分支：用 `create_upsampling_branch(num_classes)` 得到新 decoder，在 `forward` 里用 `_forward_upsample(z0,z1,z2,z3,z4, new_decoder)` 得到新图，放入 `out_dict["新head名"]`。

2. **具体修改位置（以 `cellvit.py` 为例）**
   - 在 `__init__` 中：
     - 在 `branches_output` 里增加新键（若为分割头，给出输出通道数）。
     - 若为新分割分支：`self.xxx_decoder = self.create_upsampling_branch(num_channels)`。
     - 若为新全局头：增加 `self.xxx_head = nn.Linear(...)` 或等价模块。
   - 在 `forward` 中：
     - 若是分割头：`out_dict["新head名"] = self._forward_upsample(z0,z1,z2,z3,z4, self.xxx_decoder)`。
     - 若是全局头：从 encoder 的 class token（或已有特征）过 `self.xxx_head`，结果写入 `out_dict["新head名"]`。

3. **若使用 shared 版本**：在 **`cellvit_shared.py`** 做对应修改（例如新增一个 `nn.Conv2d` 分支或一个 classifier_head）。

### 5.2 训练管线：GT、Storage、Loss

1. **数据集**（如 `monusac_patched.py`）  
   - 在 `__getitem__` 的 `masks` 字典中增加新 head 对应的 GT 键（若为分割图或回归图），保证 DataLoader 能提供该键。

2. **DataclassHVStorage**（**cellvit.py**）  
   - 在 `DataclassHVStorage` 中增加新属性（例如 `new_head: torch.Tensor`），并在 `get_dict()` 中保证该键会被返回（若需参与 loss）。

3. **Trainer**（**cell_segmentation/trainer/trainer_cellvit.py**）  
   - **unpack_predictions**：把 `predictions["新head名"]` 转成与 GT 一致的形状/类型，并填入 `DataclassHVStorage` 的新字段。  
   - **unpack_masks**：从 `masks` 中取出新 head 的 GT，转成与 pred 一致，并填入 `DataclassHVStorage`。  
   - **calculate_loss**：在 `loss_fn_dict` 中已有该 branch 的前提下，会对 `predictions.get_dict()` 里该键与 `gt.get_dict()` 里对应键求损失；因此需在 **experiment** 的 **get_loss_fn** 中为新 head 配置损失。

4. **Experiment 的 get_loss_fn**（**cell_segmentation/experiments/experiment_cellvit_pannuke.py**）  
   - 在 `get_loss_fn(self, loss_fn_settings)` 中，仿照 `nuclei_binary_map` / `hv_map` / `nuclei_type_map` / `tissue_types`，为 **新 head 名** 从 `loss_fn_settings` 里读取配置（loss_fn、weight 等），并写入 `loss_fn_dict["新head名"]`，这样 `calculate_loss` 才会对该分支做反向传播。

5. **配置文件 `train_config.yaml`**  
   - 在 **loss** 下增加新块，例如：
     ```yaml
     loss:
       # ... 现有 nuclei_binary_map, hv_map, nuclei_type_map, tissue_types ...
       新head名:
          某损失名:
            loss_fn: 已在 base_ml 注册的损失名
            weight: 1
     ```

### 5.3 推理与评估（如需要）

- 在 **inference** 模块（例如 `inference_cellvit_experiment_pannuke.py`）中，若需要保存或评估新 head 的输出，在 `unpack_predictions` / 后处理里增加对新键的处理即可。

### 5.4 小结：增加 head 的清单

| 步骤 | 位置 |
|------|------|
| 模型输出新 head | **cellvit.py**（或 **cellvit_shared.py**）：`__init__` 新 decoder/head，`forward` 写 `out_dict["新head名"]` |
| 数据集 GT | **datasets/monusac_patched.py**（或当前使用的 dataset）：`masks["新head名"]` |
| 预测/GT 容器 | **cellvit.py**：`DataclassHVStorage` 新字段 + `get_dict` |
| 训练 unpack + loss | **trainer_cellvit.py**：`unpack_predictions`、`unpack_masks`、`calculate_loss`（通过 loss_fn_dict 自动包含新 branch） |
| 损失配置 | **experiment_cellvit_pannuke.py**：`get_loss_fn` 中解析 `loss_fn_settings["新head名"]` |
| 配置项 | **train_config.yaml**：`loss.新head名` |
| 推理（可选） | **inference_cellvit_experiment_pannuke.py** 等：使用 `predictions["新head名"]` |

按上述顺序在“模型 → 数据 → Storage → Trainer → Experiment → YAML”中一致使用**同一 head 名**，即可完成“增加 head”的闭环。

---

## 6. 简要对照表

| 项目 | 说明 |
|------|------|
| **配置** | `train_config.yaml`：MoNuSAC 切块、CellViT、4 个 head 的损失与训练超参 |
| **启动** | `cell_segmentation/run_cellvit.py` → `ExperimentCellVitMoNuSAC` → `run_experiment()` + 推理 |
| **Input** | 归一化 RGB 图像，(B, 3, 256, 256) |
| **Output** | 字典：tissue_types, nuclei_binary_map, hv_map, nuclei_type_map（+ 可选 regression_map/tokens） |
| **Heads 所在文件** | 标准版：**cellvit.py**；共享 decoder 版：**cellvit_shared.py** |
| **增加 Head** | 改 cellvit/cellvit_shared → dataset → DataclassHVStorage → trainer unpack/loss → experiment get_loss_fn → train_config.yaml（及可选 inference） |

以上为对当前 `train_config.yaml`、启动项 `run_cellvit.py`、输入输出、各 head 位置以及如何增加新 head 的中文说明。
