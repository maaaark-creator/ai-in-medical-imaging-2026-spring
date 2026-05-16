# GatedDuDoRNet：第三项任务的多模态 MRI 重建模型

> 说明：本文件为中文说明文档，采用 UTF-8 编码。如果在 PowerShell 或某些旧版编辑器中看到乱码，请用支持 UTF-8 的编辑器打开，例如 VS Code、PyCharm、Typora 或新版记事本。

这个文件夹用于完成 Project 1 的第三项任务：在 T2w 欠采样重建任务中，引入 **T1n 多模态结构信息** 和 **k-space 数据一致性约束**，实现一个展开式重建网络。

本目录只负责第三项任务的高级模型部分，不依赖也不修改外层已有的第一项、第二项任务脚本。

## 1. 任务目标

第三项任务的核心要求有三点：

1. 使用展开式重建网络：把图像域去伪影网络和数据一致性层级联多次。
2. 使用多模态输入：网络同时输入欠采样 T2w 图像和全采样 T1n 图像。
3. 使用更高级的损失函数和误差分析：例如 L1 损失、SSIM 损失或混合损失，并分析重建效果最差的案例。

本目录中的模型主要完成前两点，并提供混合重建损失函数供训练脚本使用。

## 2. 模型整体思路

模型输入包括：

- `undersampled_t2`：欠采样后通过 IFFT 得到的 T2w 幅值图像，形状为 `[B, 1, H, W]`
- `t1`：同一病例、同一切片的全采样 T1n 图像，形状为 `[B, 1, H, W]`
- `mask`：k-space 欠采样 mask，形状为 `[B, 1, H, W]` 或 `[B, H, W]`
- `measured_kspace`：T2w 欠采样 k-space，形状可以是复数张量 `[B, 1, H, W]`，也可以是实部/虚部两通道 `[B, 2, H, W]`

模型输出：

- `pred_t2`：模型重建后的 T2w 图像，形状为 `[B, 1, H, W]`

整体流程如下：

```text
undersampled T2w image + fully sampled T1n
        |
        v
图像域：T1n 图像先验通过门控融合后进行网络修复
        |
        v
k-space 数据一致性约束
        |
        v
k-space 域：T1n k-space 先验通过门控融合后进行网络修复
        |
        v
k-space 数据一致性约束
        |
        v
重复多个级联模块
        |
        v
最终重建 T2w
```

## 3. 为什么使用 T1n 辅助 T2w 重建

BraTS 数据中的 T1n 和 T2w 是配准好的 3D MRI 模态。虽然二者的图像对比度不同，但它们共享很多结构信息，例如脑组织边界、肿瘤区域的大致位置、解剖结构轮廓等。

在 T2w 欠采样严重时，T2w 图像中会出现 aliasing artifact 和细节丢失。T1n 可以作为结构先验，帮助网络判断哪些边缘或结构是真实存在的，从而提升重建质量。

但是 T1n 不能被直接当成 T2w 的替代品，因为两个模态的 intensity distribution 和组织对比不同。因此这里使用 **gated fusion**，让网络自己学习在不同空间位置上应该多大程度使用 T1n 特征。

## 4. 主要文件说明

### `model.py`

包含模型主体：

- `fft2c`
  - centered orthonormal 2D FFT
  - 用于把图像域结果转回 k-space

- `ifft2c`
  - centered orthonormal 2D inverse FFT
  - 用于从 k-space 回到图像域

- `DataConsistencyLayer`
  - 在 mask 采样到的位置，用真实测量到的 k-space 替换网络预测的 k-space
  - 这是 Task 3 中 Data Consistency 的核心

- `GatedFusionUNet`
  - 图像域修复网络，对应 DuDoRNet 中的 image-domain restoration network
  - 输入为当前 T2w 图像和全采样 T1n 图像
  - DuDoRNet 原本可以使用 `concat([T2, T1])` 输入修复网络；这里改为 gated fusion，让网络学习 T1n 图像先验的使用比例

- `GatedKSpaceRefinementNet`
  - k-space 域修复网络，对应 DuDoRNet 中的 k-space-domain restoration network
  - 先将当前 T2w k-space 和 T1n k-space 都拆成实部/虚部两通道
  - DuDoRNet 原本可以使用 `concat([kT2, kT1])` 输入 k-space 修复网络；这里改为 gated fusion，控制 T1n k-space prior 注入到 T2w k-space 修复中的比例
  - 融合后通过 CNN 预测 T2w k-space residual，再加回当前 T2w k-space
  - 默认开启，可以通过 `use_kspace_refinement=False` 关闭

- `GatedDuDoRNet`
  - 最终 unrolled reconstruction model
  - 由多个 recurrent cascade 组成
  - 每个 cascade 的结构为：图像域 gated T1 fusion 修复、Data Consistency、k-space 域 gated T1 fusion 修复、第二次 Data Consistency
  - 默认 `share_cascade_weights=True`，让多个 recurrent block 共享同一组参数，更接近 DuDoRNet 的 recurrent 设定

### `losses.py`

包含训练时可用的 loss：

- `ssim_loss`
  - differentiable SSIM loss
  - 返回 `1 - SSIM`

- `HybridReconstructionLoss`
  - 默认组合为 `0.85 * L1 + 0.15 * SSIM loss`
  - 相比纯 MSE，L1 和 SSIM 通常更有利于边缘清晰度和结构保持

### `smoke_test.py`

用于检查模型能否完成一次最小前向传播。需要当前环境安装 PyTorch。

运行方式：

```powershell
cd GatedDuDoRNet
python smoke_test.py
```

如果运行成功，会打印输出 tensor 的 shape 和数值范围。

## 5. 最小使用示例

在训练脚本中可以这样使用：

```python
from GatedDuDoRNet import GatedDuDoRNet, HybridReconstructionLoss

model = GatedDuDoRNet(
    num_cascades=4,
    features=(32, 64, 128, 256),
    dc_blend=1.0,
    use_kspace_refinement=True,
    share_cascade_weights=True,
)

criterion = HybridReconstructionLoss(
    l1_weight=0.85,
    ssim_weight=0.15,
)

pred_t2 = model(
    undersampled_t2=undersampled_t2,
    t1=t1n,
    mask=mask,
    measured_kspace=measured_kspace,
    t1_kspace=t1_kspace,  # 可选；不传时模型会自动使用 fft2c(t1n)
)

loss = criterion(pred_t2, fully_sampled_t2)
```

其中：

- `undersampled_t2` 应该是 `[B, 1, H, W]`
- `t1n` 应该是 `[B, 1, H, W]`
- `mask` 应该是 `[B, 1, H, W]`
- `measured_kspace` 推荐使用 PyTorch complex tensor，形状为 `[B, 1, H, W]`
- `t1_kspace` 是可选输入，推荐预先由全采样 T1n 计算得到；如果不传，模型内部会用 `fft2c(t1n)` 计算
- `fully_sampled_t2` 是 ground truth，形状为 `[B, 1, H, W]`

## 6. Data Consistency 的重要数据要求

这一点非常关键。

严格的 Data Consistency 不能只依赖已经保存好的 magnitude undersampled image。DC 需要知道原始采样到的 k-space 点，也就是：

```text
measured_kspace = full_kspace * mask
```

因此 Task 3 的 Dataset 最好在读取 fully sampled T2w 后，实时生成欠采样数据：

```text
fully sampled T2w slice
        |
        v
FFT 得到 full k-space
        |
        v
生成或读取 variable-density mask
        |
        v
measured_kspace = full_kspace * mask
        |
        v
IFFT 得到 undersampled_t2 image
        |
        v
输入模型
```

现有的 `undersampled_raw_data_t2w_r5` 文件夹里保存的是欠采样 k-space IFFT 后的 magnitude image。它可以作为 aliased input 的来源，但因为没有保存每个 slice 对应的 `mask` 和 complex-valued `measured_kspace`，所以不适合直接用于严格 DC。

推荐做法是在 Task 3 的 dataset 中重新生成：

- `mask`
- `measured_kspace`
- `undersampled_t2`
- `fully_sampled_t2`
- `fully_sampled_t1n`

这样模型中的 `DataConsistencyLayer` 才是数学意义上正确的。

## 7. 推荐训练配置

可以从一个较轻量的配置开始：

```python
model = GatedDuDoRNet(
    num_cascades=3,
    features=(32, 64, 128),
    use_kspace_refinement=False,
)
```

如果显存足够，再升级为：

```python
model = GatedDuDoRNet(
    num_cascades=4,
    features=(32, 64, 128, 256),
    use_kspace_refinement=True,
)
```

建议训练设置：

- optimizer：Adam 或 AdamW
- learning rate：`1e-4`
- batch size：根据显存选择，常见为 `4`、`8`、`16`
- loss：`HybridReconstructionLoss`
- scheduler：`ReduceLROnPlateau` 或 cosine decay
- evaluation metrics：PSNR、SSIM

## 8. 推荐实验对比

为了写报告时更清楚地证明 Task 3 的改进，可以比较以下几组结果：

1. 欠采样输入，也就是 zero-filled / aliased reconstruction
2. Task 2 的 single-modal U-Net reconstruction
3. Task 3 的 multi-modal GatedDuDoRNet reconstruction

推荐报告指标：

- average PSNR
- average SSIM
- tissue-only PSNR / SSIM
- 5 worst cases 可视化
- residual error map

可视化图建议包含：

```text
Aliased T2w | T1n Guide | Task 2 U-Net | Task 3 GatedDuDoRNet | Ground Truth T2w | Error Map
```

## 9. 可能的消融实验

如果时间允许，可以做简单 ablation：

1. 不使用 T1n，只输入 T2w
2. 使用 T1n，但不使用 gated fusion，直接 concat
3. 使用 gated fusion，但不使用 Data Consistency
4. 使用 gated fusion + Data Consistency
5. 使用 gated fusion + Data Consistency + k-space refinement

这样可以更清楚地说明每个模块的贡献。

## 10. 当前实现的限制

当前目录已经包含模型、患者级数据划分代码和训练代码。由于本地机器性能可能不足，可以只保留代码，不在本机真正训练。

### 数据划分代码

数据划分在 `data.py` 中实现，采用患者级划分，比例为：

```text
训练集 : 验证集 : 测试集 = 7 : 1 : 2
```

划分方式不是把 NIfTI 文件移动到不同目录，而是生成一个 JSON 文件记录每个集合包含哪些病例 ID。这样不会破坏原始数据目录。

生成划分文件：

```powershell
python GatedDuDoRNet\data.py
```

默认输出：

```text
GatedDuDoRNet/splits_seed42.json
```

如果需要手动指定路径：

```powershell
python GatedDuDoRNet\data.py --archive-root archive --output-json GatedDuDoRNet/splits_seed42.json --seed 42
```

### Dataset 返回的数据

`BraTSMultiModalKSpaceDataset` 每次返回一个 2D slice，并在读取时动态生成 Task 3 需要的 k-space 数据：

```text
fully sampled T2w
→ FFT 得到 full T2w k-space
→ 生成 variable-density mask
→ measured_kspace = full_t2_kspace * mask
→ IFFT 得到 undersampled_t2
→ 同时读取 fully sampled T1n，并计算 t1_kspace
```

返回字段包括：

- `undersampled_t2`
- `t1`
- `target_t2`
- `mask`
- `measured_kspace`
- `t1_kspace`
- `case_id`
- `slice_z`

### 训练代码

训练代码在 `train.py` 中。它只负责训练和验证，不负责测试集评估。它会：

1. 读取或生成 7:1:2 患者级划分。
2. 构建 train / validation dataloader。
3. 构建 `GatedDuDoRNet` 模型。
4. 使用 `HybridReconstructionLoss` 训练。
5. 保存训练日志、最优模型权重、最后一轮模型权重和 loss 曲线。

只检查数据和模型能否构建，不开始训练：

```powershell
python GatedDuDoRNet\train.py --dry-run
```

真正训练时可以在性能更好的机器上运行：

```powershell
python GatedDuDoRNet\train.py --epochs 30 --batch-size 4 --num-cascades 4
```

默认输出目录：

```text
GatedDuDoRNet/outputs_task3/
```

其中会保存：

- `training_log.csv`
- `best_gated_dudornet.pt`
- `last_gated_dudornet_state_dict.pt`
- `loss_curve.png`

`training_log.csv` 中会记录每个 epoch 的：

- train loss
- train PSNR
- train SSIM
- validation loss
- validation PSNR
- validation SSIM
- learning rate

### 测试与评估代码

测试代码在 `test.py` 中。它会加载 `train.py` 保存的模型权重，在测试集上计算平均 loss、PSNR 和 SSIM，并随机抽取 10 张测试切片保存可视化。

默认加载：

```text
GatedDuDoRNet/outputs_task3/best_gated_dudornet.pt
```

运行测试：

```powershell
python GatedDuDoRNet\test.py
```

只检查测试集 dataloader 和模型能否构建，不加载权重、不评估：

```powershell
python GatedDuDoRNet\test.py --dry-run
```

测试脚本会保存：

- `test_metrics.txt`
- `sample_reconstructions/sample_manifest.csv`
- `sample_reconstructions/sample_*.png`

`sample_manifest.csv` 会记录每张随机样本的病例编号、slice 编号和图片路径。每张 side-by-side 可视化包含：

```text
mask | undersampled T2w | model output
```

### 仍需后续补充

后续还可以继续补充：

- testing script
- PSNR / SSIM 统计
- worst 5 cases 可视化

此外，当前模型默认输入图像已经归一化到 `[0, 1]`。如果 dataset 中使用其他归一化方式，需要保证 target、input、T1n 的尺度一致或至少稳定。

## 11. 文件结构

```text
GatedDuDoRNet/
  __init__.py
  model.py
  losses.py
  smoke_test.py
  README.md
  chat.png
```

其中 `chat.png` 是前期方案讨论截图，不参与模型运行。
