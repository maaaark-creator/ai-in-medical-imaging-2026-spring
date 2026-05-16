# BraTS MRI Reconstruction Project

## Current Layout

```text
git/
  archive/                         # BraTS data, kept outside the code repo
  ai-in-medical-imaging-2026-spring/
    task1/                          # Fourier undersampling simulation
    task2/                          # U-Net reconstruction baseline
    outputs/                        # generated results, ignored by git
```

The scripts use `--path-profile local` by default. This reads data from `../archive` and writes generated files under `outputs/`.

Use `--path-profile legacy` to keep the original cloud-platform paths/defaults. Every script also supports explicit path overrides such as `--input-root`, `--fully-sampled-dir`, `--undersampled-dir`, and `--output-dir`.

## Quick Checks

```powershell
conda activate AI_in_MI
python read_dataset_test.py
```

## Task 1

```powershell
python task1/t2w_to_kspace.py
python task1/mask.py --mode batch --save-batch-preview
python task1/prepare_undersampling_deliverables.py
```

Main local outputs:

```text
outputs/task1/kspace_t2w_slicewise_fft/
outputs/task1/undersampled_raw_data_t2w_r5/
outputs/task1/submission_r5_deliverables/
```

## Task 2

After Task 1 has generated undersampled T2w volumes:

```powershell
python task2/compute_before_recon_metrics.py --slice-grouping target_nonzero
python task2/train_unet.py
python task2/test_unet.py --slice-grouping target_nonzero
```

Main local outputs:

```text
outputs/task2/baseline_2d_unet_nonzero_train/               # 2D baseline checkpoint + split/config
outputs/task2/baseline_2d_unet_nonzero_eval_target_nonzero/ # 2D baseline evaluation outputs
outputs/task2/final_25d_unet_ctx3_nonzero_full/            # main 2.5D U-Net train + eval outputs
```
