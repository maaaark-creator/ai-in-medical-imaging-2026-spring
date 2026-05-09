from __future__ import annotations

import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_training_curves(history_csv: Path, output_dir: Path) -> None:
    history = pd.read_csv(history_csv)

    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["train_loss"], label="Train")
    if "val_loss" in history and not history["val_loss"].isna().all():
        plt.plot(history["epoch"], history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("2.5D Residual ResNet Training Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["lr"])
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "lr_curve.png", dpi=160)
    plt.close()


def save_metric_distributions(rows: list[dict[str, Any]], output_path: Path) -> None:
    before_psnr = np.array([row["before_psnr"] for row in rows], dtype=np.float32)
    after_psnr = np.array([row["after_psnr"] for row in rows], dtype=np.float32)
    before_ssim = np.array([row["before_ssim"] for row in rows], dtype=np.float32)
    after_ssim = np.array([row["after_ssim"] for row in rows], dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(before_psnr, bins=50, alpha=0.75, label="Before")
    axes[0, 0].hist(after_psnr, bins=50, alpha=0.75, label="After")
    axes[0, 0].set_title("PSNR Distribution")
    axes[0, 0].set_xlabel("PSNR (dB)")
    axes[0, 0].legend()

    axes[0, 1].hist(before_ssim, bins=50, alpha=0.75, label="Before")
    axes[0, 1].hist(after_ssim, bins=50, alpha=0.75, label="After")
    axes[0, 1].set_title("SSIM Distribution")
    axes[0, 1].set_xlabel("SSIM")
    axes[0, 1].legend()

    axes[1, 0].scatter(before_psnr, after_psnr, s=6, alpha=0.35)
    axes[1, 0].plot(
        [before_psnr.min(), before_psnr.max()],
        [before_psnr.min(), before_psnr.max()],
        "r--",
    )
    axes[1, 0].set_title("Before vs After PSNR")
    axes[1, 0].set_xlabel("Before")
    axes[1, 0].set_ylabel("After")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].scatter(before_ssim, after_ssim, s=6, alpha=0.35)
    axes[1, 1].plot(
        [before_ssim.min(), before_ssim.max()],
        [before_ssim.min(), before_ssim.max()],
        "r--",
    )
    axes[1, 1].set_title("Before vs After SSIM")
    axes[1, 1].set_xlabel("Before")
    axes[1, 1].set_ylabel("After")
    axes[1, 1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_reconstruction_samples(samples: list[dict[str, Any]], output_path: Path) -> None:
    if not samples:
        return

    rows = len(samples)
    fig, axes = plt.subplots(rows, 4, figsize=(13, 3.2 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, sample in enumerate(samples):
        input_center = sample["input_center"]
        recon = sample["recon"]
        target = sample["target"]
        error = np.abs(recon - target)
        vmax = float(np.percentile(target, 99.5)) if np.any(target) else 1.0

        panels = [
            (input_center, "Undersampled center"),
            (recon, f"2.5D ResNet\nPSNR {sample['after_psnr']:.2f} dB"),
            (target, "Ground truth"),
            (error, "Absolute error"),
        ]
        for col, (image, title) in enumerate(panels):
            cmap = "magma" if col == 3 else "gray"
            axes[row, col].imshow(
                image.T,
                cmap=cmap,
                origin="lower",
                vmin=0.0,
                vmax=vmax if col < 3 else None,
            )
            axes[row, col].set_title(title)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(f"{sample['patient_id']}\nz={sample['slice_index']}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
