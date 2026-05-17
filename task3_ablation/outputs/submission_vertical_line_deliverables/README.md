# MRI Undersampling Deliverables (Vertical-line Mask)

This folder contains organized deliverables for Fourier-domain MRI undersampling with a shared vertical-line mask.

## Settings

- Target acceleration factor: `R=5.0`
- Achieved acceleration of the shared mask: `R~5.0000`
- Number of exported visualization examples: `5`
- Number of standalone metric samples: `24`

## Folder layout

- `01_mask/`: the shared vertical-line undersampling mask in PNG and NPY format
- `02_comparisons/`: side-by-side comparison figures showing mask, fully sampled image, and aliased image
- `03_image_pairs/`: fully sampled vs aliased image pairs for the same slices
- `examples_manifest.csv`: case and slice metadata for the exported figures
- `example_metrics.csv` / `example_metrics.json`: PSNR/SSIM/MAE/RMSE for the exported figure examples
- `sample_metrics.csv` / `sample_metrics.json`: PSNR/SSIM/MAE/RMSE for a larger representative slice subset
