# Task 2: 2.5D Residual ResNet Reconstruction

This folder contains a reproducible T2-only 2.5D residual ResNet pipeline for reconstructing R=5 undersampled BraTS T2w images.

The model predicts a correction residual and returns:

```text
reconstruction = undersampled_center_slice + predicted_residual
```

The main Task 2 loss is MSE/L2, matching the assignment requirement.

## Experiments

Experiment A reproduces the existing per-slice independent normalization style:

```powershell
python train.py --normalization independent --context-slices 3 --slice-filter nonzero --batch-size 4 --epochs 30
python evaluate.py --normalization independent --model-path ..\outputs\task2\exp_25d_resnet_ctx3_nonzero_independent_train\best_resnet25d.pth --output-dir ..\outputs\task2\exp_25d_resnet_ctx3_nonzero_independent_train
```

Experiment B uses shared normalization from the fully sampled T2 volume:

```powershell
python train.py --normalization shared --context-slices 3 --slice-filter nonzero --batch-size 4 --epochs 30
python evaluate.py --normalization shared --model-path ..\outputs\task2\exp_25d_resnet_ctx3_nonzero_shared_train\best_resnet25d.pth --output-dir ..\outputs\task2\exp_25d_resnet_ctx3_nonzero_shared_train
```

If another training job is still running and you want a safer one-click evaluation after it finishes:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_shared_eval.ps1 -WaitForNoPython
```

## Smoke Tests

```powershell
python train.py --normalization independent --limit-patients 3 --epochs 1 --batch-size 1 --num-workers 0 --base-channels 16 --num-blocks 2 --output-dir ..\outputs\task2\25d_resnet_smoke_independent
python evaluate.py --normalization independent --model-path ..\outputs\task2\25d_resnet_smoke_independent\best_resnet25d.pth --output-dir ..\outputs\task2\25d_resnet_smoke_independent --batch-size 1 --num-workers 0 --num-samples 3

python train.py --normalization shared --limit-patients 3 --epochs 1 --batch-size 1 --num-workers 0 --base-channels 16 --num-blocks 2 --output-dir ..\outputs\task2\25d_resnet_smoke_shared
python evaluate.py --normalization shared --model-path ..\outputs\task2\25d_resnet_smoke_shared\best_resnet25d.pth --output-dir ..\outputs\task2\25d_resnet_smoke_shared --batch-size 1 --num-workers 0 --num-samples 3
```

## Outputs

Default outputs are written under `outputs/task2/`:

- `best_resnet25d.pth`, `last_checkpoint.pth`
- `config.json`, `split_patients.json`, `dataset_stats.json`
- `history.csv`, `loss_curve.png`, `lr_curve.png`
- `metrics.txt`, `metrics.json`, `per_slice_metrics.csv`
- `before_after_comparison.csv`
- `metric_distributions.png`, `reconstruction_samples.png`

## Blank Slice Policy

Training defaults to `--slice-filter nonzero --blank-threshold 0.001`, which removes near-blank center slices based on the fully sampled target. Evaluation defaults to `--slice-filter all` and reports both all slices and non-blank/tissue slices.

## Normalization Modes

- `independent`: each input slice and target slice is min-max normalized separately.
- `shared`: all input context slices and the target slice share the same patient-level scale, computed as the 99.5 percentile of nonzero fully sampled T2 voxels.
