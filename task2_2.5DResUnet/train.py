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
    DEFAULT_BASE_CHANNELS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BLANK_THRESHOLD,
    DEFAULT_CACHE_SIZE,
    DEFAULT_CONTEXT_SLICES,
    DEFAULT_DILATIONS,
    DEFAULT_EARLY_STOP_PATIENCE,
    DEFAULT_EPOCHS,
    DEFAULT_FULLY_SAMPLED_DIR,
    DEFAULT_LR,
    DEFAULT_NORMALIZATION,
    DEFAULT_NUM_BLOCKS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SEED,
    DEFAULT_SLICE_FILTER,
    DEFAULT_UNDERSAMPLED_DIR,
    DEFAULT_WEIGHT_DECAY,
    default_output_dir,
    expand_path,
    list_common_patients,
    parse_dilations,
    save_json,
    set_seed,
    split_patients,
)
from dataset import BraTS25DResNetSliceDataset, validate_context_slices, validate_normalization
from model import ResNet25D
from visualize import save_training_curves

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 2.5D residual ResNet for BraTS T2w reconstruction.")
    parser.add_argument("--undersampled-dir", type=Path, default=DEFAULT_UNDERSAMPLED_DIR)
    parser.add_argument("--fully-sampled-dir", type=Path, default=DEFAULT_FULLY_SAMPLED_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--context-slices", type=int, default=DEFAULT_CONTEXT_SLICES)
    parser.add_argument("--slice-filter", choices=["all", "nonzero"], default=DEFAULT_SLICE_FILTER)
    parser.add_argument("--normalization", choices=["independent", "shared"], default=DEFAULT_NORMALIZATION)
    parser.add_argument("--blank-threshold", type=float, default=DEFAULT_BLANK_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--cache-size", type=int, default=DEFAULT_CACHE_SIZE)
    parser.add_argument("--limit-patients", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=DEFAULT_BASE_CHANNELS)
    parser.add_argument("--num-blocks", type=int, default=DEFAULT_NUM_BLOCKS)
    parser.add_argument("--dilations", type=str, default=DEFAULT_DILATIONS)
    return parser.parse_args()


def make_loader(
    dataset: BraTS25DResNetSliceDataset,
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
    validate_normalization(args.normalization)
    dilations = parse_dilations(args.dilations)
    set_seed(args.seed)

    undersampled_dir = expand_path(args.undersampled_dir)
    fully_sampled_dir = expand_path(args.fully_sampled_dir)
    output_dir = expand_path(
        args.output_dir
        or default_output_dir(args.context_slices, args.slice_filter, args.normalization)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device             : {device}")
    print(f"Undersampled root  : {undersampled_dir}")
    print(f"Fully sampled root : {fully_sampled_dir}")
    print(f"Output dir         : {output_dir}")
    print(f"Normalization      : {args.normalization}")

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
            "dilations_tuple": dilations,
        }
    )
    save_json(output_dir / "config.json", config_payload)
    save_json(output_dir / "split_patients.json", split)

    train_dataset = BraTS25DResNetSliceDataset(
        split["train"],
        undersampled_dir,
        fully_sampled_dir,
        context_slices=args.context_slices,
        slice_filter=args.slice_filter,
        normalization=args.normalization,
        blank_threshold=args.blank_threshold,
        cache_size=args.cache_size,
        desc="Indexing train",
    )
    val_dataset = BraTS25DResNetSliceDataset(
        split["val"],
        undersampled_dir,
        fully_sampled_dir,
        context_slices=args.context_slices,
        slice_filter=args.slice_filter,
        normalization=args.normalization,
        blank_threshold=args.blank_threshold,
        cache_size=args.cache_size,
        desc="Indexing val",
    )
    test_stats_dataset = BraTS25DResNetSliceDataset(
        split["test"],
        undersampled_dir,
        fully_sampled_dir,
        context_slices=args.context_slices,
        slice_filter=args.slice_filter,
        normalization=args.normalization,
        blank_threshold=args.blank_threshold,
        cache_size=0,
        desc="Indexing test stats",
    )
    save_json(
        output_dir / "dataset_stats.json",
        {
            "train": train_dataset.stats(),
            "val": val_dataset.stats(),
            "test": test_stats_dataset.stats(),
        },
    )
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

    model = ResNet25D(
        in_channels=args.context_slices,
        base_channels=args.base_channels,
        num_blocks=args.num_blocks,
        dilations=dilations,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
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
            "base_channels": args.base_channels,
            "num_blocks": args.num_blocks,
            "dilations": dilations,
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
            torch.save(checkpoint, output_dir / "best_resnet25d.pth")
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
            "best_checkpoint": str(output_dir / "best_resnet25d.pth"),
            "history_csv": str(output_dir / "history.csv"),
        },
    )
    print(f"Training complete. Best checkpoint: {output_dir / 'best_resnet25d.pth'}")


if __name__ == "__main__":
    main()
