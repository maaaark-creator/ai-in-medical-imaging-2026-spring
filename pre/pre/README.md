# Task 2 presentation assets

This folder is for generating figures and tables used in the Task 2 presentation.

## Run

From the workspace root:

```powershell
python git/pre/build_task2_presentation_assets.py
```

All outputs are written to:

```text
git/pre/assets/
```

## Main generated assets

- `task2_main_results.csv/json/md`  
  Main tissue-slice result table for Input, 2D U-Net, 2.5D U-Net, and 2.5D residual ResNet.

- `main_results_psnr_ssim.png`  
  Bar chart for tissue PSNR and SSIM.

- `consistent_tissue_metric_distributions.png`  
  Unified PSNR/SSIM distribution plot for Input, 2D U-Net, 2.5D U-Net, and 2.5D ResNet. This replaces the inconsistent per-model distribution figures for presentation use.

- `psnr_gain_over_input.png`  
  PSNR gain over the aliased input. Useful for the main conclusion slide.

- `line_mask_pipeline_example.png`  
  Real k-space example showing vertical-line mask, fully sampled target, aliased input, and absolute aliasing error.

- `real_line_input_target_pair.png`  
  Real NIfTI input/target pair from the line-mask dataset. This uses the lightweight NIfTI reader inside the script and does not require `nibabel`.

- `kspace_masking_demo.png`  
  Full k-space, line mask, and masked k-space visualization.

- `vertical_line_mask_r5_seed42.png`  
  Standalone vertical-line R=5 mask.

- `pointwise_vs_line_mask_comparison.png`  
  Preliminary point-wise mask versus final vertical-line mask.

- `loss_curves_contact_sheet.png`  
  Contact sheet of existing 2D U-Net, 2.5D U-Net, and 2.5D ResNet loss curves.

- `combined_25d_training_curves.png`  
  Train/validation curves for the two 2.5D models using their `history.csv` files.

- `reconstruction_samples_*.png`  
  Copied qualitative reconstruction samples from existing experiment outputs.

- `25d_*_worst_tissue_slices.csv` and `25d_*_best_gain_tissue_slices.csv`  
  Useful backup tables for error analysis or teacher questions.

- `resnet_worst5_tissue_visuals.png` and `resnet_worst5_tissue_visual_cases.csv`  
  Worst 5 tissue slices for the final 2.5D residual ResNet by after-PSNR, with aliased input, reconstruction, ground truth, and absolute error. This is most useful as a backup/error-analysis slide.

- `presentation_notes.md`  
  Short interpretation and caveats for the presentation.

- `asset_manifest.json`  
  List of generated assets and file sizes.

## Suggested slide use

- Objective / pipeline: `line_mask_pipeline_example.png`
- Why line mask: `vertical_line_mask_r5_seed42.png`, `real_line_input_target_pair.png`, or `kspace_masking_demo.png`
- Training setup: `loss_curves_contact_sheet.png`
- Quantitative results: `task2_main_results.md`, `main_results_psnr_ssim.png`, `psnr_gain_over_input.png`, `consistent_tissue_metric_distributions.png`
- Qualitative results: `reconstruction_samples_25d_resnet.png`
- Preliminary exploration: `pointwise_vs_line_mask_comparison.png`
- Limitations / Q&A: `presentation_notes.md`, worst-slice CSV files, and `resnet_worst5_tissue_visuals.png`

## Notes

The script does not require `nibabel`; it reads existing k-space `.npz` files, experiment outputs, and simple uncompressed or gzipped NIfTI-1 files through a lightweight local reader. If the wider 2.5D U-Net result is added later, update the experiment section in `build_task2_presentation_assets.py` or add a separate row to `task2_main_results.csv`.
