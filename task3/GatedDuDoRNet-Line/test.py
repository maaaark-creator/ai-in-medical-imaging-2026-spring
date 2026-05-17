from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_: object):
        return iterable

from data import BraTSMultiModalKSpaceDataset, load_split_json
from losses import HybridReconstructionLoss
from model import GatedDuDoRNet


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
DEFAULT_ARCHIVE_ROOT = REPO_ROOT / "archive"
DEFAULT_UNDERSAMPLED_ROOT = REPO_ROOT / "undersampled_raw_data_t2w_vertical_line_r5"
DEFAULT_MASKED_KSPACE_ROOT = REPO_ROOT / "masked_kspace_t2w_vertical_line_r5"
DEFAULT_SPLIT_JSON = MODULE_DIR / "splits_seed42.json"
DEFAULT_OUTPUT_DIR = MODULE_DIR / "outputs_task3_line"
DEFAULT_CHECKPOINT = DEFAULT_OUTPUT_DIR / "best_gated_dudornet.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained GatedDuDoRNet checkpoint on the test set.")
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--undersampled-root", type=Path, default=DEFAULT_UNDERSAMPLED_ROOT)
    parser.add_argument("--masked-kspace-root", type=Path, default=DEFAULT_MASKED_KSPACE_ROOT)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--acceleration", type=float, default=5.0)
    parser.add_argument("--center-fraction", type=float, default=0.10)
    parser.add_argument("--sigma", type=float, default=0.28)
    parser.add_argument("--num-cascades", type=int, default=4)
    parser.add_argument("--residual-scale", type=float, default=0.10)
    parser.add_argument("--no-kspace-refinement", action="store_true")
    parser.add_argument("--no-shared-cascade-weights", action="store_true")
    parser.add_argument("--num-sample-images", type=int, default=10)
    parser.add_argument(
        "--blank-zero-fraction",
        type=float,
        default=0.999,
        help="Exclude test slices whose fully sampled T2w image has at least this fraction of zero pixels.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build test loader/model and exit before evaluation.")
    return parser.parse_args()


def remove_blank_slices(dataset: BraTSMultiModalKSpaceDataset, zero_fraction_threshold: float) -> int:
    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "nibabel is required to filter blank test slices."
        ) from exc

    kept_slice_index: list[tuple[int, int]] = []
    removed = 0

    for case_idx, (_, _, t2_path, _, num_slices) in enumerate(dataset.cases):
        t2_img = nib.load(str(t2_path))
        for slice_z in range(num_slices):
            t2_slice = np.asarray(t2_img.dataobj[:, :, slice_z])
            zero_fraction = 1.0 - (np.count_nonzero(t2_slice) / t2_slice.size)
            if zero_fraction >= zero_fraction_threshold:
                removed += 1
                continue
            kept_slice_index.append((case_idx, slice_z))

    dataset.slice_index = kept_slice_index
    return removed


def build_test_loader(args: argparse.Namespace) -> DataLoader:
    splits = load_split_json(args.split_json)
    dataset = BraTSMultiModalKSpaceDataset(
        archive_root=args.archive_root,
        undersampled_root=args.undersampled_root,
        masked_kspace_root=args.masked_kspace_root,
        case_ids=splits["test"],
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        sigma=args.sigma,
        seed=args.seed,
    )
    removed_blank_slices = remove_blank_slices(dataset, args.blank_zero_fraction)
    print(
        f"Excluded blank test slices: {removed_blank_slices} "
        f"(zero pixel fraction >= {args.blank_zero_fraction:.4f})"
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def move_batch_to_device(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    tensor_keys = [
        "undersampled_t2",
        "t1",
        "target_t2",
        "mask",
        "measured_kspace",
        "t1_kspace",
    ]
    return {key: batch[key].to(device, non_blocking=True) for key in tensor_keys}


def batch_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((pred.clamp(0.0, 1.0) - target) ** 2, dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-12))


def batch_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred.clamp(0.0, 1.0) - target.clamp(0.0, 1.0)), dim=(1, 2, 3))


def batch_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred.clamp(0.0, 1.0) - target.clamp(0.0, 1.0)) ** 2, dim=(1, 2, 3))


def batch_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(batch_mse(pred, target), min=0.0))


def batch_ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    c1 = 0.01**2
    c2 = 0.03**2
    kernel = torch.ones((1, 1, 7, 7), device=pred.device, dtype=pred.dtype) / 49.0

    mu_x = F.conv2d(pred, kernel, padding=3)
    mu_y = F.conv2d(target, kernel, padding=3)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sigma_x = F.conv2d(pred * pred, kernel, padding=3) - mu_x2
    sigma_y = F.conv2d(target * target, kernel, padding=3) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel, padding=3) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim_map.mean(dim=(1, 2, 3))


def load_model(args: argparse.Namespace, device: torch.device) -> GatedDuDoRNet:
    model = GatedDuDoRNet(
        num_cascades=args.num_cascades,
        features=(32, 64, 128, 256),
        use_kspace_refinement=not args.no_kspace_refinement,
        share_cascade_weights=not args.no_shared_cascade_weights,
        residual_scale=args.residual_scale,
    ).to(device)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return model


def evaluate(
    model: GatedDuDoRNet,
    loader: DataLoader,
    criterion: HybridReconstructionLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    metric_values: dict[str, list[float]] = {
        "mae": [],
        "mse": [],
        "rmse": [],
        "psnr": [],
        "ssim": [],
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc="test"):
            batch_t = move_batch_to_device(batch, device)
            pred = model(
                undersampled_t2=batch_t["undersampled_t2"],
                t1=batch_t["t1"],
                mask=batch_t["mask"],
                measured_kspace=batch_t["measured_kspace"],
                t1_kspace=batch_t["t1_kspace"],
            )
            loss = criterion(pred, batch_t["target_t2"])
            batch_size = batch_t["target_t2"].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            metric_values["mae"].extend(batch_mae(pred, batch_t["target_t2"]).cpu().tolist())
            metric_values["mse"].extend(batch_mse(pred, batch_t["target_t2"]).cpu().tolist())
            metric_values["rmse"].extend(batch_rmse(pred, batch_t["target_t2"]).cpu().tolist())
            metric_values["psnr"].extend(batch_psnr(pred, batch_t["target_t2"]).cpu().tolist())
            metric_values["ssim"].extend(batch_ssim(pred, batch_t["target_t2"]).cpu().tolist())

    total_samples = max(total_samples, 1)
    results: dict[str, float] = {
        "test_loss": total_loss / total_samples,
        "num_samples": float(len(metric_values["mae"])),
    }
    for metric_name, values in metric_values.items():
        if values:
            arr = np.asarray(values, dtype=np.float64)
            results[f"{metric_name}_mean"] = float(arr.mean())
            results[f"{metric_name}_var"] = float(arr.var())
            results[f"{metric_name}_std"] = float(arr.std())
        else:
            results[f"{metric_name}_mean"] = 0.0
            results[f"{metric_name}_var"] = 0.0
            results[f"{metric_name}_std"] = 0.0
    return results


def save_random_reconstruction_samples(
    model: GatedDuDoRNet,
    dataset,
    device: torch.device,
    output_dir: Path,
    num_samples: int,
    seed: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping reconstruction samples.")
        return

    sample_dir = output_dir / "sample_reconstructions"
    sample_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    sample_indices = rng.sample(range(len(dataset)), k=min(num_samples, len(dataset)))
    csv_path = sample_dir / "sample_manifest.csv"

    model.eval()
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "dataset_index", "case_id", "slice_z", "figure_path"])

        with torch.no_grad():
            for sample_i, dataset_idx in enumerate(sample_indices, start=1):
                item = dataset[dataset_idx]
                batch = {
                    key: item[key].unsqueeze(0).to(device)
                    for key in [
                        "undersampled_t2",
                        "t1",
                        "target_t2",
                        "mask",
                        "measured_kspace",
                        "t1_kspace",
                    ]
                }
                pred = model(
                    undersampled_t2=batch["undersampled_t2"],
                    t1=batch["t1"],
                    mask=batch["mask"],
                    measured_kspace=batch["measured_kspace"],
                    t1_kspace=batch["t1_kspace"],
                )

                mask = batch["mask"].cpu().squeeze().numpy()
                undersampled = batch["undersampled_t2"].cpu().squeeze().numpy()
                output = pred.cpu().squeeze().numpy()
                case_id = str(item["case_id"])
                slice_z = int(item["slice_z"])

                fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
                axes[0].imshow(mask, cmap="gray")
                axes[0].set_title("Mask")
                axes[1].imshow(undersampled, cmap="gray", vmin=0.0, vmax=1.0)
                axes[1].set_title("Undersampled T2w")
                axes[2].imshow(output, cmap="gray", vmin=0.0, vmax=1.0)
                axes[2].set_title("Model Output")
                for ax in axes:
                    ax.axis("off")
                fig.suptitle(f"{case_id} | slice {slice_z}")
                fig.tight_layout()

                figure_name = f"sample_{sample_i:02d}_{case_id}_slice_{slice_z:03d}.png"
                figure_path = sample_dir / figure_name
                fig.savefig(figure_path, dpi=180)
                plt.close(fig)
                writer.writerow([sample_i, dataset_idx, case_id, slice_z, str(figure_path)])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = build_test_loader(args)
    print(f"Test slices: {len(test_loader.dataset)}")

    model = GatedDuDoRNet(
        num_cascades=args.num_cascades,
        features=(32, 64, 128, 256),
        use_kspace_refinement=not args.no_kspace_refinement,
        share_cascade_weights=not args.no_shared_cascade_weights,
        residual_scale=args.residual_scale,
    ).to(device)

    if args.dry_run:
        print("Dry run finished. Model and test dataloader were built; no checkpoint was loaded.")
        return

    model = load_model(args, device)
    criterion = HybridReconstructionLoss()
    metrics = evaluate(model, test_loader, criterion, device)

    with (args.output_dir / "test_metrics.txt").open("w", encoding="utf-8") as f:
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"num_samples: {int(metrics['num_samples'])}\n")
        f.write(f"test_loss: {metrics['test_loss']:.8f}\n")
        for metric_name in ["mae", "mse", "rmse", "psnr", "ssim"]:
            f.write(f"{metric_name}_mean: {metrics[f'{metric_name}_mean']:.8f}\n")
            f.write(f"{metric_name}_var: {metrics[f'{metric_name}_var']:.8f}\n")
            f.write(f"{metric_name}_std: {metrics[f'{metric_name}_std']:.8f}\n")

    save_random_reconstruction_samples(
        model=model,
        dataset=test_loader.dataset,
        device=device,
        output_dir=args.output_dir,
        num_samples=args.num_sample_images,
        seed=args.seed,
    )
    print(
        " | ".join(
            [
                f"Test loss: {metrics['test_loss']:.6f}",
                f"MAE mean/std: {metrics['mae_mean']:.6f}/{metrics['mae_std']:.6f}",
                f"MSE mean/std: {metrics['mse_mean']:.6f}/{metrics['mse_std']:.6f}",
                f"RMSE mean/std: {metrics['rmse_mean']:.6f}/{metrics['rmse_std']:.6f}",
                f"PSNR mean/std: {metrics['psnr_mean']:.3f}/{metrics['psnr_std']:.3f}",
                f"SSIM mean/std: {metrics['ssim_mean']:.4f}/{metrics['ssim_std']:.4f}",
            ]
        )
    )


if __name__ == "__main__":
    main()
