# Task 2 presentation materials

This folder contains the Task 2 presentation deliverables, generated figures,
tables, notes, and scripts.

## Layout

- `scripts/`: scripts used to generate or repair presentation files.
- `presentation/`: final presentation exports, including the PDF.
- `package/`: bundled asset archive.
- `contrast/`: contrast sample set and manifest.
- `assets/figures/`: generated plots, visual examples, training curves, and reconstruction samples.
- `assets/tables/`: CSV and Markdown result tables.
- `assets/notes/`: interpretation notes and caveats.
- `assets/metadata/`: JSON metadata and manifests.
- `assets/model_diagrams/`: model architecture diagrams.

## Run

From the workspace root:

```powershell
python git/pre/scripts/build_task2_presentation_assets.py
```

All presentation outputs are written back into:

```text
git/pre/assets/
```

For the extended nonzero-slice MAE/RMSE tables:

```powershell
python git/pre/scripts/compute_extended_nonzero_metrics.py
```

## Key Files

- `presentation/Task 2 MRI Reconstruction Presentation.pdf`
- `assets/tables/task2_main_results.csv`
- `assets/tables/task2_main_results.md`
- `assets/metadata/task2_main_results.json`
- `assets/figures/main_results_psnr_ssim.png`
- `assets/figures/consistent_tissue_metric_distributions.png`
- `assets/figures/psnr_gain_over_input.png`
- `assets/figures/task2_nonzero_key_metrics.png`
- `assets/figures/line_mask_pipeline_example.png`
- `assets/figures/real_line_input_target_pair.png`
- `assets/figures/pointwise_vs_line_mask_comparison.png`
- `assets/notes/presentation_notes.md`
- `assets/metadata/asset_manifest.json`

## Notes

The scripts read experiment outputs and sibling workspace data directories, but
all generated presentation materials are kept under this `git/pre/` folder.
