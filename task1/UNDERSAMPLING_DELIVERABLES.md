# MRI 欠采样结果使用说明

这份文档说明了如何生成并整理本项目中 MRI 傅里叶域欠采样实验所需的结果文件，方便别人直接复现和使用。

本流程对应的主要任务包括：

- 生成一个二维随机 variable-density 欠采样 mask，采样加速因子为 `R=5`
- 将全采样 T2 图像通过 FFT 转换到 k-space
- 在 k-space 中施加欠采样 mask
- 通过 IFFT 重建 aliased / undersampled 图像
- 导出用于提交的对比图和图像对

## 1. 相关脚本

本项目里和这个流程相关的主要脚本有：

- [t2w_to_kspace.py](t2w_to_kspace.py)
  
  作用：把原始全采样 T2w 体数据逐 slice 做 2D FFT，生成 k-space 文件。

- [mask.py](mask.py)
  
  作用：生成 variable-density mask，在 k-space 中做欠采样，并通过 IFFT 重建 aliased 图像。

- [prepare_undersampling_deliverables.py](prepare_undersampling_deliverables.py)
  
  作用：把结果整理成适合提交的目录结构，自动导出 mask 图、三联对比图和 full/aliased 图像对。

## 2. 输入数据目录要求

当前本地整理后的推荐结构是：代码仓库和数据集同级放在 `git/` 下，脚本默认使用 `--path-profile local`，也就是从仓库外一层的 `../archive/` 读取数据：

```text
git/
  archive/
    BraTS-GLI-00000-000/
      BraTS-GLI-00000-000-t2w.nii
      BraTS-GLI-00000-000-t1c.nii
      ...
  ai-in-medical-imaging-2026-spring/
    task1/
    task2/
```

如果需要复用原始教学/云平台写法，也可以加 `--path-profile legacy`，此时会回到旧的 `raw_data/` 相对路径约定：

```text
raw_data/
  BraTS-GLI-00000-000/
    BraTS-GLI-00000-000-t2w.nii
    BraTS-GLI-00000-000-t1c.nii
    ...
  BraTS-GLI-00001-000/
    BraTS-GLI-00001-000-t2w.nii
    ...
```

在这个欠采样流程里，实际会用到的是每个病例下的：

- `*-t2w.nii`
- 或 `*-t2w.nii.gz`

## 3. 整个流程在做什么

### 第一步：把全采样图像转换到 k-space

运行命令：

```powershell
python task1/t2w_to_kspace.py
```

这个脚本会对每个 T2w volume 的每一张 slice 做 2D FFT，并保存为 `.npz` 文件。

默认输出位置：

```text
outputs/task1/kspace_t2w_slicewise_fft/
  BraTS-GLI-00000-000/
    BraTS-GLI-00000-000-t2w_kspace_complex.npz
```

同时还会生成一张 k-space 预览图，保存在：

```text
outputs/task1/kspace_previews/
```

### 第二步：生成 mask 并模拟欠采样、混叠

如果只想先看一张示例图，可以运行：

```powershell
python task1/mask.py --mode preview --acceleration 5
```

这个命令会生成：

- 一个二维随机 variable-density 欠采样 mask
- 一张全采样 slice
- 一张对应的 aliased / undersampled 重建图
- 一张三联对比图

默认输出位置：

```text
outputs/task1/undersampling_preview/
  variable_density_mask_r5_preview.png
  variable_density_mask_r5.npy
  *_undersampling_preview.png
```

如果想把所有病例都生成欠采样后的体数据，可以运行：

```powershell
python task1/mask.py --mode batch --acceleration 5 --save-batch-preview
```

默认输出位置：

```text
outputs/task1/undersampled_raw_data_t2w_r5/
  BraTS-GLI-00000-000/
    BraTS-GLI-00000-000-t2w.nii
```

这里保存的是欠采样重建后的 T2w 体数据。

### 第三步：整理成可提交的结果目录

如果你想直接得到一套适合提交或汇报展示的结果，运行：

```powershell
python task1/prepare_undersampling_deliverables.py
```

默认输出位置：

```text
outputs/task1/submission_r5_deliverables/
  01_mask/
  02_comparisons/
  03_image_pairs/
  examples_manifest.csv
  README.md
```

## 4. 推荐直接使用的命令

如果已经准备好当前的 `../archive/` 数据目录，而且只需要导出 Task 1 展示材料，那么最推荐直接运行：

```powershell
python task1/prepare_undersampling_deliverables.py
```

这个命令会自动完成以下事情：

- 生成一个共享的二维随机 variable-density mask
- 使用 `R=5` 作为加速因子
- 默认导出 `5` 组示例
- 保存 mask 的 PNG 和 NPY
- 保存 `5` 张三联对比图
- 保存 `5` 组 full / aliased 图像对
- 保存一个 CSV 清单，记录每组图来自哪个病例和哪一层 slice

## 5. 常用参数说明

### 修改导出的示例数量

```powershell
python task1/prepare_undersampling_deliverables.py --num-examples 8
```

### 修改随机种子

```powershell
python task1/prepare_undersampling_deliverables.py --seed 123
```

### 修改输出目录

```powershell
python task1/prepare_undersampling_deliverables.py --output-dir outputs/my_submission_set
```

### 修改加速因子

```powershell
python task1/prepare_undersampling_deliverables.py --acceleration 5
```

如果是当前这次作业要求，建议保持为 `5`。

## 6. 输出目录说明

运行 `prepare_undersampling_deliverables.py` 后，主要输出目录含义如下：

- `01_mask/`
  
  保存本次示例使用的 Fourier 域欠采样 mask。

- `02_comparisons/`
  
  保存三联图，每张图包含：
  
  - sampling mask
  - fully sampled image
  - aliased / undersampled image

- `03_image_pairs/`
  
  保存两联图，每张图包含：
  
  - fully sampled image
  - aliased / undersampled image

- `examples_manifest.csv`
  
  保存每张图对应的 `case_id` 和 `slice_index`。

- `README.md`
  
  保存这次导出结果的简要说明。

## 7. 当前已经生成好的结果

目前已经整理好的提交目录在：

- `outputs/task1/submission_r5_deliverables`

其中比较关键的文件有：

- `outputs/task1/submission_r5_deliverables/01_mask/variable_density_mask_r5.png`
- `outputs/task1/submission_r5_deliverables/examples_manifest.csv`

示例三联图：

- `outputs/task1/submission_r5_deliverables/02_comparisons/*_comparison.png`

示例 full / aliased 图像对：

- `outputs/task1/submission_r5_deliverables/03_image_pairs/*_pair.png`

## 8. 完整运行流程

如果从头开始运行，推荐按下面顺序执行：

```powershell
python task1/t2w_to_kspace.py
python task1/mask.py --mode preview --acceleration 5
python task1/mask.py --mode batch --acceleration 5 --save-batch-preview
python task1/prepare_undersampling_deliverables.py
```

如果前面的 k-space 和欠采样体数据都已经生成好了，通常直接运行下面这一句就够了：

```powershell
python task1/prepare_undersampling_deliverables.py
```

## 9. 实验含义说明

这个流程的核心思想是：

1. 先把空间域的 MRI 图像变换到傅里叶域，也就是 k-space。
2. 在 k-space 中用一个随机 variable-density mask 只保留一部分采样点。
3. 因为采样不完整，直接做 IFFT 回到图像域后，就会产生混叠和细节损失。
4. 通过对比全采样图像和 aliased 图像，可以直观看到欠采样带来的伪影和质量下降。

## 10. 备注

- 当前默认使用固定随机种子，因此结果是可复现的。
- 当前默认加速因子为 `R=5`，符合本次任务要求。
- 欠采样是在二维 k-space 上进行的。
- aliased 图像是通过“全采样 k-space × 欠采样 mask”后，再做 IFFT 得到的。
