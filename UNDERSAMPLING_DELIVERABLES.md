# MRI Undersampling Deliverables Guide

This document explains how to generate and organize the MRI undersampling results required for submission:

- a 2D random variable-density undersampling mask with acceleration factor `R=5`
- fully sampled T2 image slices
- undersampled / aliased reconstructions from Fourier-domain masking
- side-by-side visualization figures
- fully-sampled / aliased image pairs

## 1. Project files

The main scripts involved in this workflow are:

- [t2w_to_kspace.py](/d:/ai_medical_imaging/project1/t2w_to_kspace.py): converts fully sampled T2w volumes into slice-wise k-space with FFT
- [mask.py](/d:/ai_medical_imaging/project1/mask.py): creates variable-density masks, simulates undersampling, and reconstructs aliased images
- [prepare_undersampling_deliverables.py](/d:/ai_medical_imaging/project1/prepare_undersampling_deliverables.py): exports a clean submission-ready folder with mask, comparison figures, and image pairs

## 2. Expected input data layout

The scripts assume the original BraTS-style dataset is stored like this:

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

Only the `*-t2w.nii` or `*-t2w.nii.gz` files are used in this undersampling workflow.

## 3. What each step does

### Step A: Convert fully sampled images to k-space

This uses 2D FFT on each axial slice:

```powershell
python t2w_to_kspace.py
```

Default output:

```text
kspace_t2w_slicewise_fft/
  BraTS-GLI-00000-000/
    BraTS-GLI-00000-000-t2w_kspace_complex.npz
```

It also saves a preview image in:

```text
outputs/kspace_previews/
```

### Step B: Simulate undersampling and aliasing

To generate a single preview figure with:

- sampling mask
- fully sampled slice
- aliased slice

run:

```powershell
python mask.py --mode preview --acceleration 5
```

Default preview output:

```text
outputs/undersampling_preview/
  variable_density_mask_r5_preview.png
  variable_density_mask_r5.npy
  *_undersampling_preview.png
```

To reconstruct all T2w cases into undersampled / aliased volumes:

```powershell
python mask.py --mode batch --acceleration 5 --save-batch-preview
```

Default batch output:

```text
undersampled_raw_data_t2w_r5/
  BraTS-GLI-00000-000/
    BraTS-GLI-00000-000-t2w.nii
```

### Step C: Export a submission-ready deliverables folder

This is the simplest command if you want a clean set of figures for homework or report submission:

```powershell
python prepare_undersampling_deliverables.py
```

Default output:

```text
outputs/submission_r5_deliverables/
  01_mask/
  02_comparisons/
  03_image_pairs/
  examples_manifest.csv
  README.md
```

## 4. Recommended command for this project

If the FFT files and original `raw_data/` already exist, the main command you usually need is:

```powershell
python prepare_undersampling_deliverables.py
```

This will:

- create one shared 2D random variable-density mask
- use `R=5`
- export `5` example slices by default
- save one mask PNG and one mask NPY
- save `5` comparison figures with `mask + fully sampled + aliased`
- save `5` fully-sampled / aliased image pairs
- write a CSV manifest listing case IDs and slice indices

## 5. Useful options

### Change the number of exported examples

```powershell
python prepare_undersampling_deliverables.py --num-examples 8
```

### Change the random seed

```powershell
python prepare_undersampling_deliverables.py --seed 123
```

### Change the output folder

```powershell
python prepare_undersampling_deliverables.py --output-dir outputs/my_submission_set
```

### Change the acceleration factor

```powershell
python prepare_undersampling_deliverables.py --acceleration 5
```

For this assignment, keep it at `5` unless the requirement changes.

## 6. Output folder explanation

After running `prepare_undersampling_deliverables.py`, the output folders mean:

- `01_mask/`: the Fourier-domain undersampling mask used for the examples
- `02_comparisons/`: 3-panel figures showing sampling mask, fully sampled image, and aliased image
- `03_image_pairs/`: 2-panel figures showing fully sampled image and aliased image only
- `examples_manifest.csv`: which case and slice each exported figure came from
- `README.md`: a short summary of the exported deliverables

## 7. Current generated results

The latest prepared submission folder is:

- [outputs/submission_r5_deliverables](/d:/ai_medical_imaging/project1/outputs/submission_r5_deliverables)

Its main contents include:

- [variable_density_mask_r5.png](/d:/ai_medical_imaging/project1/outputs/submission_r5_deliverables/01_mask/variable_density_mask_r5.png)
- [examples_manifest.csv](/d:/ai_medical_imaging/project1/outputs/submission_r5_deliverables/examples_manifest.csv)

Example comparison figure:

- [BraTS-GLI-00000-000_slice_077_comparison.png](/d:/ai_medical_imaging/project1/outputs/submission_r5_deliverables/02_comparisons/BraTS-GLI-00000-000_slice_077_comparison.png)

Example image pair:

- [BraTS-GLI-00000-000_slice_077_pair.png](/d:/ai_medical_imaging/project1/outputs/submission_r5_deliverables/03_image_pairs/BraTS-GLI-00000-000_slice_077_pair.png)

## 8. Typical workflow summary

For a fresh run, the full workflow is:

```powershell
python t2w_to_kspace.py
python mask.py --mode preview --acceleration 5
python mask.py --mode batch --acceleration 5 --save-batch-preview
python prepare_undersampling_deliverables.py
```

If the k-space data and undersampled data have already been generated, you can usually skip directly to:

```powershell
python prepare_undersampling_deliverables.py
```

## 9. Notes

- The undersampling mask is generated in 2D k-space.
- Aliasing is created by multiplying the full k-space with the mask and reconstructing with inverse FFT.
- The aliased images are expected to look blurrier and contain undersampling artifacts compared with the fully sampled images.
- The deliverables script uses a fixed random seed by default so the exported examples are reproducible.
