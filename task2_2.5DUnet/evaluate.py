from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BLANK_THRESHOLD,
    DEFAULT_CACHE_SIZE,
    DEFAULT_CONTEXT_SLICES,
    DEFAULT_FULLY_SAMPLED_DIR,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SEED,
    DEFAULT_UNDERSAMPLED_DIR,
    default_output_dir,
    expand_path,
    list_common_patients,
    load_json,
    save_json,
    set_seed,
    split_patients,
)
from dataset import BraTS25DSliceDataset, validate_context_slices
from metrics import compute_psnr_ssim, summarize_rows, write_metrics_csv, write_summary_text
from model import UNet25D
from visualize import save_metric_distributions, save_reconstruction_samples

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained 2.5D U-Net reconstruction model.")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--undersampled-dir", type=Path, default=DEFAULT_UNDERSAMPLED_DIR)
    parser.add_argument("--fully-sampled-dir", type=Path, default=DEFAULT_FULLY_SAMPLED_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--context-slices", type=int, default=DEFAULT_CONTEXT_SLICES)
    parser.add_argument("--slice-filter", choices=["all", "nonzero"], default="all")
    parser.add_argument("--blank-threshold", type=float, default=DEFAULT_BLANK_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--cache-size", type=int, default=DEFAULT_CACHE_SIZE)
    parser.add_argument("--limit-patients", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=6)
    return parser.parse_args()


def load_checkpoint(model_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint
    return {"model_state": checkpoint, "config": {}, "split": None, "features": (32, 64, 128, 256)}


def make_loader(dataset: BraTS25DSliceDataset, args: argparse.Namespace, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def maybe_store_sample(
    reservoir: list[dict[str, Any]],
    candidate: dict[str, Any],
    seen: int,
    capacity: int,
    rng: random.Random,
) -> None:
    if capacity <= 0:
        return
    if len(reservoir) < capacity:
        reservoir.append(candidate)
        return
    replace_idx = rng.randint(0, seen)
    if replace_idx < capacity:
        reservoir[replace_idx] = candidate


def main() -> None:
    args = parse_args()
    validate_context_slices(args.context_slices)
    set_seed(args.seed)

    output_dir = expand_path(args.output_dir or default_output_dir(args.context_slices, "nonzero"))
    model_path = expand_path(args.model_path or output_dir / "best_unet25d.pth")
    undersampled_dir = expand_path(args.undersampled_dir)
    fully_sampled_dir = expand_path(args.fully_sampled_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(model_path, device)
    checkpoint_config = checkpoint.get("config", {})
    features = tuple(checkpoint.get("features", (32, 64, 128, 256)))

    context_slices = int(checkpoint_config.get("context_slices", args.context_slices))
    validate_context_slices(context_slices)
    model = UNet25D(in_channels=context_slices, features=features).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    split = checkpoint.get("split")
    split_file = output_dir / "split_patients.json"
    if split is None and split_file.exists():
        split = load_json(split_file)
    if split is None:
        common_patients = list_common_patients(undersampled_dir, fully_sampled_dir)
        split = split_patients(common_patients, seed=args.seed, limit_patients=args.limit_patients)
    elif args.limit_patients is not None:
        split = {key: value[: args.limit_patients] for key, value in split.items()}

    test_patients = split.get("test", [])
    if not test_patients:
        print("No held-out test patients found; using validation patients for smoke-test evaluation.")
        test_patients = split.get("val", [])
    if not test_patients:
        raise RuntimeError("No test or validation patients are available for evaluation.")

    dataset = BraTS25DSliceDataset(
        test_patients,
        undersampled_dir,
        fully_sampled_dir,
        context_slices=context_slices,
        slice_filter=args.slice_filter,
        blank_threshold=args.blank_threshold,
        cache_size=args.cache_size,
        return_metadata=True,
        desc="Indexing eval",
    )
    if len(dataset) == 0:
        raise RuntimeError("No evaluation slices were available after filtering.")

    save_json(output_dir / "eval_dataset_stats.json", dataset.stats())
    loader = make_loader(dataset, args, device)
    rows: list[dict[str, Any]] = []
    sample_reservoir: list[dict[str, Any]] = []
    sample_rng = random.Random(args.seed)
    seen_nonblank = 0

    with torch.no_grad():
        for inputs, targets, metadata in tqdm(loader, desc="Evaluating"):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)

            inputs_np = inputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            outputs_np = outputs.cpu().numpy()
            center_inputs = metadata["center_input"].numpy()

            patient_ids = metadata["patient_id"]
            slice_indices = metadata["slice_index"].numpy()
            fractions = metadata["target_nonzero_fraction"].numpy()

            for i in range(inputs_np.shape[0]):
                target = targets_np[i, 0]
                recon = outputs_np[i, 0]
                input_center = center_inputs[i, 0]
                before_psnr, before_ssim = compute_psnr_ssim(input_center, target)
                after_psnr, after_ssim = compute_psnr_ssim(recon, target)
                frac = float(fractions[i])
                is_nonblank = frac >= args.blank_threshold
                row = {
                    "patient_id": patient_ids[i],
                    "slice_index": int(slice_indices[i]),
                    "target_nonzero_fraction": frac,
                    "is_nonblank": is_nonblank,
                    "before_psnr": before_psnr,
                    "before_ssim": before_ssim,
                    "after_psnr": after_psnr,
                    "after_ssim": after_ssim,
                }
                rows.append(row)

                if is_nonblank:
                    candidate = {
                        "patient_id": patient_ids[i],
                        "slice_index": int(slice_indices[i]),
                        "input_center": input_center,
                        "recon": np.clip(recon, 0.0, 1.0),
                        "target": target,
                        "before_psnr": before_psnr,
                        "after_psnr": after_psnr,
                    }
                    maybe_store_sample(
                        sample_reservoir,
                        candidate,
                        seen_nonblank,
                        args.num_samples,
                        sample_rng,
                    )
                    seen_nonblank += 1

    summary = summarize_rows(rows, args.blank_threshold)
    save_json(output_dir / "metrics.json", summary)
    write_summary_text(output_dir / "metrics.txt", summary)
    write_metrics_csv(output_dir / "per_slice_metrics.csv", rows)
    save_metric_distributions(rows, output_dir / "metric_distributions.png")
    save_reconstruction_samples(sample_reservoir, output_dir / "reconstruction_samples.png")

    comparison_rows = []
    for key, values in summary.items():
        comparison_rows.append(
            {
                "group": key,
                "count": values["count"],
                "before_psnr": values["before_psnr"],
                "after_psnr": values["after_psnr"],
                "psnr_gain": values["after_psnr"] - values["before_psnr"],
                "before_ssim": values["before_ssim"],
                "after_ssim": values["after_ssim"],
                "ssim_gain": values["after_ssim"] - values["before_ssim"],
            }
        )
    import csv

    with (output_dir / "before_after_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"Evaluation complete. Metrics: {output_dir / 'metrics.txt'}")


if __name__ == "__main__":
    main()

