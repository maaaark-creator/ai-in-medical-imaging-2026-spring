from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_: object):
        return iterable

from data import BraTSMultiModalKSpaceDataset, load_split_json, save_split_json, SplitConfig
from losses import HybridReconstructionLoss
from model import GatedDuDoRNet


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
DEFAULT_ARCHIVE_ROOT = REPO_ROOT / "archive"
DEFAULT_UNDERSAMPLED_ROOT = REPO_ROOT / "undersampled_raw_data_t2w_vertical_line_r5"
DEFAULT_MASKED_KSPACE_ROOT = REPO_ROOT / "masked_kspace_t2w_vertical_line_r5"
DEFAULT_SPLIT_JSON = MODULE_DIR / "splits_seed42.json"
DEFAULT_OUTPUT_DIR = MODULE_DIR / "outputs_task3_line"
LOSS_KEYS = ["total_loss", "l1_loss", "ssim_loss", "weighted_l1", "weighted_ssim"]


def serializable_args(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training code for Task 3 GatedDuDoRNet. This script is provided for running on a stronger machine."
    )
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--undersampled-root", type=Path, default=DEFAULT_UNDERSAMPLED_ROOT)
    parser.add_argument("--masked-kspace-root", type=Path, default=DEFAULT_MASKED_KSPACE_ROOT)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--acceleration", type=float, default=5.0)
    parser.add_argument("--center-fraction", type=float, default=0.10)
    parser.add_argument("--sigma", type=float, default=0.28)
    parser.add_argument("--num-cascades", type=int, default=4)
    parser.add_argument("--no-kspace-refinement", action="store_true")
    parser.add_argument("--no-shared-cascade-weights", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build datasets/model and exit before training.")
    return parser.parse_args()


def ensure_splits(args: argparse.Namespace) -> dict[str, list[str]]:
    if args.split_json.exists():
        return load_split_json(args.split_json)

    config = SplitConfig(
        archive_root=str(args.archive_root),
        output_json=str(args.split_json),
        seed=args.seed,
    )
    return save_split_json(config)


def build_loader(
    archive_root: Path,
    case_ids: list[str],
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    dataset = BraTSMultiModalKSpaceDataset(
        archive_root=archive_root,
        undersampled_root=args.undersampled_root,
        masked_kspace_root=args.masked_kspace_root,
        case_ids=case_ids,
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        sigma=args.sigma,
        seed=args.seed,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
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


def run_epoch(
    model: GatedDuDoRNet,
    loader: DataLoader,
    criterion: HybridReconstructionLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    total_epochs: int | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    loss_sums = {key: 0.0 for key in LOSS_KEYS}
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    phase = "train" if is_train else "val"
    desc = phase
    if epoch is not None and total_epochs is not None:
        desc = f"Epoch {epoch:03d}/{total_epochs:03d} {phase}"
    with context:
        for batch in tqdm(loader, desc=desc):
            batch_t = move_batch_to_device(batch, device)
            pred = model(
                undersampled_t2=batch_t["undersampled_t2"],
                t1=batch_t["t1"],
                mask=batch_t["mask"],
                measured_kspace=batch_t["measured_kspace"],
                t1_kspace=batch_t["t1_kspace"],
            )
            loss_parts = criterion.components(pred, batch_t["target_t2"])
            loss = loss_parts["total_loss"]

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = batch_t["target_t2"].size(0)
            for key in LOSS_KEYS:
                loss_sums[key] += loss_parts[key].item() * batch_size
            with torch.no_grad():
                total_psnr += batch_psnr(pred.detach(), batch_t["target_t2"]).sum().item()
                total_ssim += batch_ssim(pred.detach(), batch_t["target_t2"]).sum().item()
            total_samples += batch_size

    total_samples = max(total_samples, 1)
    metrics = {key: value / total_samples for key, value in loss_sums.items()}
    metrics["psnr"] = total_psnr / total_samples
    metrics["ssim"] = total_ssim / total_samples
    return metrics


def plot_loss_curve(history: list[dict[str, float]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping loss curve.")
        return

    epochs = [item["epoch"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    val_losses = [item["val_loss"] for item in history]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_losses, label="Train Loss", marker="o", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Validation Loss", marker="o", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Task 3 Training Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = ensure_splits(args)
    print("Patient-level split:")
    for name, ids in splits.items():
        print(f"  {name}: {len(ids)} cases")

    train_loader = build_loader(args.archive_root, splits["train"], args, shuffle=True)
    val_loader = build_loader(args.archive_root, splits["val"], args, shuffle=False)
    print(f"Train slices: {len(train_loader.dataset)}")
    print(f"Val slices:   {len(val_loader.dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GatedDuDoRNet(
        num_cascades=args.num_cascades,
        features=(32, 64, 128, 256),
        use_kspace_refinement=not args.no_kspace_refinement,
        share_cascade_weights=not args.no_shared_cascade_weights,
    ).to(device)
    criterion = HybridReconstructionLoss(l1_weight=0.85, ssim_weight=0.15)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    if args.dry_run:
        print("Dry run finished. Model and dataloaders were built; no training was started.")
        return

    best_val_loss = float("inf")
    history: list[dict[str, float]] = []
    log_path = args.output_dir / "training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_l1_loss",
                "train_ssim_loss",
                "train_weighted_l1",
                "train_weighted_ssim",
                "train_psnr",
                "train_ssim",
                "val_loss",
                "val_l1_loss",
                "val_ssim_loss",
                "val_weighted_l1",
                "val_weighted_ssim",
                "val_psnr",
                "val_ssim",
                "lr",
            ]
        )

        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                epoch=epoch,
                total_epochs=args.epochs,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                epoch=epoch,
                total_epochs=args.epochs,
            )
            train_loss = train_metrics["total_loss"]
            val_loss = val_metrics["total_loss"]
            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]["lr"]
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.8f}",
                    f"{train_metrics['l1_loss']:.8f}",
                    f"{train_metrics['ssim_loss']:.8f}",
                    f"{train_metrics['weighted_l1']:.8f}",
                    f"{train_metrics['weighted_ssim']:.8f}",
                    f"{train_metrics['psnr']:.6f}",
                    f"{train_metrics['ssim']:.6f}",
                    f"{val_loss:.8f}",
                    f"{val_metrics['l1_loss']:.8f}",
                    f"{val_metrics['ssim_loss']:.8f}",
                    f"{val_metrics['weighted_l1']:.8f}",
                    f"{val_metrics['weighted_ssim']:.8f}",
                    f"{val_metrics['psnr']:.6f}",
                    f"{val_metrics['ssim']:.6f}",
                    f"{lr:.8e}",
                ]
            )
            f.flush()
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "train_l1_loss": train_metrics["l1_loss"],
                    "train_ssim_loss": train_metrics["ssim_loss"],
                    "train_weighted_l1": train_metrics["weighted_l1"],
                    "train_weighted_ssim": train_metrics["weighted_ssim"],
                    "train_psnr": train_metrics["psnr"],
                    "train_ssim": train_metrics["ssim"],
                    "val_loss": val_loss,
                    "val_l1_loss": val_metrics["l1_loss"],
                    "val_ssim_loss": val_metrics["ssim_loss"],
                    "val_weighted_l1": val_metrics["weighted_l1"],
                    "val_weighted_ssim": val_metrics["weighted_ssim"],
                    "val_psnr": val_metrics["psnr"],
                    "val_ssim": val_metrics["ssim"],
                }
            )

            print(
                f"\nEpoch {epoch:03d}/{args.epochs:03d} summary\n"
                f"  train total={train_loss:.6f} | "
                f"l1={train_metrics['l1_loss']:.6f} (w={train_metrics['weighted_l1']:.6f}) | "
                f"ssim_loss={train_metrics['ssim_loss']:.6f} (w={train_metrics['weighted_ssim']:.6f}) | "
                f"psnr={train_metrics['psnr']:.3f} | metric_ssim={train_metrics['ssim']:.4f}\n"
                f"  val   total={val_loss:.6f} | "
                f"l1={val_metrics['l1_loss']:.6f} (w={val_metrics['weighted_l1']:.6f}) | "
                f"ssim_loss={val_metrics['ssim_loss']:.6f} (w={val_metrics['weighted_ssim']:.6f}) | "
                f"psnr={val_metrics['psnr']:.3f} | metric_ssim={val_metrics['ssim']:.4f}\n"
                f"  lr:    {lr:.2e}"
            )

            if val_loss < best_val_loss:
                previous_best = best_val_loss
                best_val_loss = val_loss
                checkpoint_path = args.output_dir / "best_gated_dudornet.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "args": serializable_args(args),
                    },
                    checkpoint_path,
                )
                if previous_best == float("inf"):
                    print(f"  best model: initialized at val_loss={val_loss:.6f}")
                else:
                    print(
                        f"  best model: improved from {previous_best:.6f} to {val_loss:.6f}"
                    )
                print(f"  saved checkpoint: {checkpoint_path}")
            else:
                print(
                    f"  best model: not improved; best val_loss remains {best_val_loss:.6f}"
                )

    torch.save(model.state_dict(), args.output_dir / "last_gated_dudornet_state_dict.pt")
    plot_loss_curve(history, args.output_dir)
    print("Training finished.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Run test.py to evaluate the saved model on the test set.")


if __name__ == "__main__":
    main()
