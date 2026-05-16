# MRI Undersampling Deliverables

This folder contains organized deliverables for Fourier-domain MRI undersampling with a random variable-density mask.

## Settings

- Target acceleration factor: `R=5.0`
- Achieved acceleration of the shared 2D mask: `R~5.0000`
- Number of exported examples: `1`

## Folder layout

- `01_mask/`: the shared undersampling mask in PNG and NPY format
- `02_comparisons/`: side-by-side comparison figures showing mask, fully sampled image, and aliased image
- `03_image_pairs/`: fully sampled vs aliased image pairs for the same slices
- `examples_manifest.csv`: case and slice metadata for each exported example
