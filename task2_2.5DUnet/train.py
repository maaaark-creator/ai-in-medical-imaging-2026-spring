from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BLANK_THRESHOLD,
    DEFAULT_CACHE_SIZE,
    DEFAULT_CONTEXT_SLICES,
    DEFAULT_EARLY_STOP_PATIENCE,
    DEFAULT_EPOCHS,
    DEFAULT_FULLY_SAMPLED_DIR,
    DEFAULT_LR,
    DEFAULT_NORM_MODE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_ROBUST_PERCENTILE,
    DEFAULT_SEED,
    DEFAULT_SLICE_FILTER,
    DEFAULT_UNDERSAMPLED_DIR,
    DEFAULT_WEIGHT_DECAY,
    default_output_dir,
    expand_path,
    list_common_patients,
    save_json,
    set_seed,
    split_patients,
)
from dataset import BraTS25DSliceDataset, validate_context_slices
from model import UNet25D
from visualize import save_training_curves

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 2.5D U-Net for BraTS T2w reconstruction.")
    parser.add_argument("--undersampled-dir", type=Path, default=DEFAULT_UNDERSAMPLED_DIR)
    parser.add_argument("--fully-sampled-dir", type=Path, default=DEFAULT_FULLY_SAMPLED_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--context-slices", type=int, default=DEFAULT_CONTEXT_SLICES)
    parser.add_argument("--slice-filter", choices=["all", "nonzero"], default=DEFAULT_SLICE_FILTER)
    parser.add_argument("--blank-threshold", type=float, default=DEFAULT_BLANK_THRESHOLD)
    parser.add_argument(
        "--norm-mode",
        choices=["separate", "target-volume-robust"],
        default=DEFAULT_NORM_MODE,
        help="separate: old per-slice independent min-max; target-volume-robust: input/target share target volume pXX scale.",
    )
    parser.add_argument("--robust-percentile", type=float, default=DEFAULT_ROBUST_PERCENTILE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--cache-size", type=int, default=DEFAULT_CACHE_SIZE)
    parser.add_argument("--limit-patients", type=int, default=None)
    parser.add_argument("--base-features", type=int, default=32)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a checkpoint such as last_checkpoint.pth.",
    )
    return parser.parse_args()


def make_loader(
    dataset: BraTS25DSliceDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    desc: str = "Epoch",
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_items = 0

    for inputs, targets in tqdm(loader, desc=desc):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = inputs.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


def write_history(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "monitor_loss", "lr", "seconds"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_context_slices(args.context_slices)
    set_seed(args.seed)

    undersampled_dir = expand_path(args.undersampled_dir)
    fully_sampled_dir = expand_path(args.fully_sampled_dir)
    output_dir = expand_path(
        args.output_dir or default_output_dir(args.context_slices, args.slice_filter, args.norm_mode)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device             : {device}")
    print(f"Undersampled root  : {undersampled_dir}")
    print(f"Fully sampled root : {fully_sampled_dir}")
    print(f"Output dir         : {output_dir}")

    common_patients = list_common_patients(undersampled_dir, fully_sampled_dir)
    split = split_patients(common_patients, seed=args.seed, limit_patients=args.limit_patients)
    print(
        f"Patients           : train={len(split['train'])}, "
        f"val={len(split['val'])}, test={len(split['test'])}"
    )

    config_payload = vars(args).copy()
    config_payload.update(
        {
            "undersampled_dir": str(undersampled_dir),
            "fully_sampled_dir": str(fully_sampled_dir),
            "output_dir": str(output_dir),
            "device": str(device),
        }
    )
    save_json(output_dir / "config.json", config_payload)
    save_json(output_dir / "split_patients.json", split)

    train_dataset = BraTS25DSliceDataset(
        split["train"],
        undersampled_dir,
        fully_sampled_dir,
        context_slices=args.context_slices,
        slice_filter=args.slice_filter,
        blank_threshold=args.blank_threshold,
        norm_mode=args.norm_mode,
        robust_percentile=args.robust_percentile,
        cache_size=args.cache_size,
        desc="Indexing train",
    )
    val_dataset = BraTS25DSliceDataset(
        split["val"],
        undersampled_dir,
        fully_sampled_dir,
        context_slices=args.context_slices,
        slice_filter=args.slice_filter,
        blank_threshold=args.blank_threshold,
        norm_mode=args.norm_mode,
        robust_percentile=args.robust_percentile,
        cache_size=args.cache_size,
        desc="Indexing val",
    )
    test_stats_dataset = BraTS25DSliceDataset(
        split["test"],
        undersampled_dir,
        fully_sampled_dir,
        context_slices=args.context_slices,
        slice_filter=args.slice_filter,
        blank_threshold=args.blank_threshold,
        norm_mode=args.norm_mode,
        robust_percentile=args.robust_percentile,
        cache_size=0,
        desc="Indexing test stats",
    )
    dataset_stats = {
        "train": train_dataset.stats(),
        "val": val_dataset.stats(),
        "test": test_stats_dataset.stats(),
    }
    save_json(output_dir / "dataset_stats.json", dataset_stats)
    print(f"Train slices       : {len(train_dataset)}")
    print(f"Val slices         : {len(val_dataset)}")

    if len(train_dataset) == 0:
        raise RuntimeError("No training slices were available after filtering.")

    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, device)
    val_loader = (
        make_loader(val_dataset, args.batch_size, False, args.num_workers, device)
        if len(val_dataset) > 0
        else None
    )

    features = tuple(args.base_features * (2**i) for i in range(4))
    model = UNet25D(in_channels=args.context_slices, features=features).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    start_epoch = 1

    if args.resume is not None:
        resume_path = expand_path(args.resume)
        checkpoint = torch.load(resume_path, map_location=device)
        checkpoint_features = tuple(checkpoint.get("features", features))
        if checkpoint_features != features:
            raise ValueError(
                f"Checkpoint features {checkpoint_features} do not match current features {features}. "
                "Use the same --base-features value as the original run."
            )
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            for group in optimizer.param_groups:
                group["lr"] = min(float(group["lr"]), args.lr)
        best_loss = float(checkpoint.get("best_loss", best_loss))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

        history_path = output_dir / "history.csv"
        if history_path.exists():
            import csv as csv_module

            with history_path.open("r", newline="", encoding="utf-8") as f:
                history = [
                    {
                        "epoch": int(row["epoch"]),
                        "train_loss": float(row["train_loss"]),
                        "val_loss": float(row["val_loss"]),
                        "monitor_loss": float(row["monitor_loss"]),
                        "lr": float(row["lr"]),
                        "seconds": float(row["seconds"]),
                    }
                    for row in csv_module.DictReader(f)
                    if int(row["epoch"]) < start_epoch
                ]
        print(f"Resumed from {resume_path} at epoch {start_epoch}. Best loss: {best_loss:.8f}")

    if start_epoch > args.epochs:
        print(f"Checkpoint already reached epoch {start_epoch - 1}; nothing to train.")

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            desc=f"Epoch {epoch}/{args.epochs} [train]",
        )
        val_loss = (
            run_epoch(model, val_loader, criterion, device, desc=f"Epoch {epoch}/{args.epochs} [val]")
            if val_loader is not None
            else float("nan")
        )
        monitor_loss = val_loss if val_loader is not None else train_loss
        scheduler.step(monitor_loss)
        lr = float(optimizer.param_groups[0]["lr"])
        seconds = time.time() - start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "monitor_loss": monitor_loss,
            "lr": lr,
            "seconds": seconds,
        }
        history.append(row)
        write_history(output_dir / "history.csv", history)

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_loss": best_loss,
            "config": config_payload,
            "split": split,
            "features": features,
        }
        torch.save(checkpoint, output_dir / "last_checkpoint.pth")

        print(
            f"Epoch {epoch:03d}: train={train_loss:.6f}, "
            f"val={val_loss:.6f}, monitor={monitor_loss:.6f}, lr={lr:.2e}"
        )

        if monitor_loss < best_loss:
            best_loss = monitor_loss
            epochs_without_improvement = 0
            checkpoint["best_loss"] = best_loss
            torch.save(checkpoint, output_dir / "best_unet25d.pth")
            print("  -> Saved best checkpoint")
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement for {epochs_without_improvement} epoch(s)")
            if epochs_without_improvement >= args.early_stop_patience:
                print("Early stopping triggered.")
                break

    save_training_curves(output_dir / "history.csv", output_dir)
    save_json(
        output_dir / "training_summary.json",
        {
            "best_loss": best_loss,
            "epochs_ran": len(history),
            "best_checkpoint": str(output_dir / "best_unet25d.pth"),
            "history_csv": str(output_dir / "history.csv"),
        },
    )
    print(f"Training complete. Best checkpoint: {output_dir / 'best_unet25d.pth'}")


if __name__ == "__main__":
    main()
