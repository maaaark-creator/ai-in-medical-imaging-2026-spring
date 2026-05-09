# Task 2: 2.5D U-Net Reconstruction

This folder contains a clean 2.5D U-Net pipeline for BraTS T2w MRI reconstruction from R=5 undersampled images.

## Recommended Commands

Small CPU smoke test:

```powershell
D:\conda_data\envs\AI_in_MI\python.exe train.py --limit-patients 3 --epochs 1 --batch-size 1 --num-workers 0 --base-features 8 --output-dir ..\outputs\task2\25d_unet_smoke
D:\conda_data\envs\AI_in_MI\python.exe evaluate.py --model-path ..\outputs\task2\25d_unet_smoke\best_unet25d.pth --output-dir ..\outputs\task2\25d_unet_smoke --batch-size 1 --num-workers 0 --num-samples 3
```

Full GPU run:

```powershell
python train.py --context-slices 3 --slice-filter nonzero --batch-size 8 --epochs 30
python evaluate.py --model-path ..\outputs\task2\25d_unet_context3_nonzero\best_unet25d.pth --output-dir ..\outputs\task2\25d_unet_context3_nonzero
```

Use 5 neighboring slices by changing both commands to `--context-slices 5`.

## Outputs

Main outputs are written to `outputs/task2/25d_unet_context3_nonzero` by default:

- `best_unet25d.pth`: best checkpoint by validation loss.
- `history.csv`, `loss_curve.png`, `lr_curve.png`: training monitoring.
- `dataset_stats.json`, `eval_dataset_stats.json`: slice filtering and patient loading stats.
- `metrics.txt`, `metrics.json`: before/after PSNR and SSIM for all and non-blank slices.
- `per_slice_metrics.csv`: per-slice PSNR/SSIM before and after reconstruction.
- `metric_distributions.png`, `reconstruction_samples.png`: report-ready visualizations.

## Blank Slice Policy

The default training setting is `--slice-filter nonzero`, which removes near-blank center slices using the fully sampled target slice. A slice is treated as non-blank when:

```text
target_nonzero_fraction >= 0.001
```

Evaluation defaults to `--slice-filter all` and reports both all slices and non-blank/tissue slices, so the report stays transparent while the main comparison is not inflated by easy blank slices.

