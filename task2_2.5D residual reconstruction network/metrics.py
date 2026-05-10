from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr_ssim(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    pred = np.asarray(pred, dtype=np.float32).squeeze()
    target = np.asarray(target, dtype=np.float32).squeeze()
    pred = np.clip(pred, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)

    mse = float(np.mean((pred - target) ** 2))
    if mse == 0.0:
        psnr = 100.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            psnr = float(peak_signal_noise_ratio(target, pred, data_range=1.0))
        if np.isinf(psnr):
            psnr = 100.0

    if float(np.std(pred)) < 1e-8 or float(np.std(target)) < 1e-8:
        ssim = 1.0 if np.allclose(pred, target) else 0.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ssim = float(structural_similarity(target, pred, data_range=1.0))

    return psnr, ssim


def summarize_rows(rows: list[dict[str, Any]], blank_threshold: float) -> dict[str, Any]:
    def mean_of(key: str, selected: list[dict[str, Any]]) -> float:
        return float(np.mean([row[key] for row in selected])) if selected else 0.0

    nonblank = [
        row for row in rows if float(row["target_nonzero_fraction"]) >= blank_threshold
    ]
    return {
        "all_slices": {
            "count": len(rows),
            "before_psnr": mean_of("before_psnr", rows),
            "before_ssim": mean_of("before_ssim", rows),
            "after_psnr": mean_of("after_psnr", rows),
            "after_ssim": mean_of("after_ssim", rows),
        },
        "nonblank_slices": {
            "count": len(nonblank),
            "blank_threshold": blank_threshold,
            "before_psnr": mean_of("before_psnr", nonblank),
            "before_ssim": mean_of("before_ssim", nonblank),
            "after_psnr": mean_of("after_psnr", nonblank),
            "after_ssim": mean_of("after_ssim", nonblank),
        },
    }


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patient_id",
        "slice_index",
        "target_nonzero_fraction",
        "is_nonblank",
        "before_psnr",
        "before_ssim",
        "after_psnr",
        "after_ssim",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_text(path: Path, summary: dict[str, Any]) -> None:
    lines = ["=== 2.5D Residual ResNet Reconstruction Metrics ===", ""]
    for section_name, values in summary.items():
        title = "All slices" if section_name == "all_slices" else "Non-blank / tissue slices"
        lines.append(f"[{title}]")
        lines.append(f"Count       : {values['count']}")
        if "blank_threshold" in values:
            lines.append(f"Threshold   : target nonzero fraction >= {values['blank_threshold']}")
        lines.append(f"Before PSNR : {values['before_psnr']:.4f} dB")
        lines.append(f"Before SSIM : {values['before_ssim']:.6f}")
        lines.append(f"After PSNR  : {values['after_psnr']:.4f} dB")
        lines.append(f"After SSIM  : {values['after_ssim']:.6f}")
        lines.append(f"PSNR gain   : {values['after_psnr'] - values['before_psnr']:.4f} dB")
        lines.append(f"SSIM gain   : {values['after_ssim'] - values['before_ssim']:.6f}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
