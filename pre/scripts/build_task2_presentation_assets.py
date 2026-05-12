"""
Build presentation-ready assets for Task 2.

Run from the workspace root:
    python git/pre/scripts/build_task2_presentation_assets.py

Outputs are written to:
    git/pre/assets/

The script is intentionally self-contained and does not require nibabel. It uses
existing experiment outputs plus k-space NPZ files that are already in git/.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import gzip
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - optional for asset generation
    torch = None
    nn = None


SCRIPT_DIR = Path(__file__).resolve().parent
PRE_DIR = SCRIPT_DIR.parent
GIT_DIR = PRE_DIR.parent
ASSET_DIR = PRE_DIR / "assets"
FIGURE_DIR = ASSET_DIR / "figures"
TABLE_DIR = ASSET_DIR / "tables"
NOTE_DIR = ASSET_DIR / "notes"
METADATA_DIR = ASSET_DIR / "metadata"

LINE_REPO = GIT_DIR / "ai-in-medical-imaging-2026-spring -linemask"
OLD_REPO = GIT_DIR / "ai-in-medical-imaging-2026-spring"
LINE_TASK2 = LINE_REPO / "outputs" / "task2"
ARCHIVE_ROOT = GIT_DIR / "archive"
LINE_UNDERSAMPLED_ROOT = GIT_DIR / "undersampled_raw_data_t2w_vertical_line_r5"

EXPERIMENTS = {
    "2D U-Net": {
        "kind": "2d",
        "train_dir": LINE_TASK2 / "baseline_2d_unet_vertical_line_r5_all_p99_train",
        "eval_dir": LINE_TASK2 / "baseline_2d_unet_vertical_line_r5_all_p99_eval_target_nonzero",
    },
    "2.5D U-Net": {
        "kind": "25d_unet",
        "dir": LINE_TASK2 / "final_25d_unet_vertical_line_r5_ctx3_all_p99_full",
    },
    "2.5D U-Net bf64": {
        "kind": "25d_unet_bf64",
        "dir": LINE_TASK2 / "final_25d_unet_vertical_line_r5_ctx3_all_p99_bf64_full",
    },
    "2.5D Residual ResNet": {
        "kind": "25d_resnet",
        "dir": LINE_TASK2 / "exp_25d_resnet_vertical_line_r5_ctx3_all_shared_p99_train",
    },
}


def ensure_dirs() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for directory in (FIGURE_DIR, TABLE_DIR, NOTE_DIR, METADATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_before_after_csv(path: Path, group: str) -> dict:
    df = pd.read_csv(path)
    row = df.loc[df["group"] == group].iloc[0]
    return row.to_dict()


def build_results_table() -> pd.DataFrame:
    resnet_dir = EXPERIMENTS["2.5D Residual ResNet"]["dir"]
    resnet_tissue = read_before_after_csv(resnet_dir / "before_after_comparison.csv", "nonblank_slices")
    resnet_all = read_before_after_csv(resnet_dir / "before_after_comparison.csv", "all_slices")

    unet25_dir = EXPERIMENTS["2.5D U-Net"]["dir"]
    unet25_tissue = read_before_after_csv(unet25_dir / "before_after_comparison.csv", "nonblank_slices")
    unet25_all = read_before_after_csv(unet25_dir / "before_after_comparison.csv", "all_slices")

    unet25_bf64_dir = EXPERIMENTS["2.5D U-Net bf64"]["dir"]
    unet25_bf64_tissue = read_before_after_csv(unet25_bf64_dir / "before_after_comparison.csv", "nonblank_slices")
    unet25_bf64_all = read_before_after_csv(unet25_bf64_dir / "before_after_comparison.csv", "all_slices")

    unet2_summary = load_json(EXPERIMENTS["2D U-Net"]["eval_dir"] / "metrics_summary.json")
    unet2_tissue = unet2_summary["selected_slices"]
    unet2_all = unet2_summary["all_slices"]

    before_tissue_psnr = float(resnet_tissue["before_psnr"])
    before_tissue_ssim = float(resnet_tissue["before_ssim"])
    before_all_psnr = float(resnet_all["before_psnr"])
    before_all_ssim = float(resnet_all["before_ssim"])

    rows = [
        {
            "method": "Input / before recon",
            "tissue_psnr": before_tissue_psnr,
            "tissue_ssim": before_tissue_ssim,
            "psnr_gain": np.nan,
            "ssim_gain": np.nan,
            "all_psnr": before_all_psnr,
            "all_ssim": before_all_ssim,
        },
        {
            "method": "2D U-Net",
            "tissue_psnr": float(unet2_tissue["psnr"]),
            "tissue_ssim": float(unet2_tissue["ssim"]),
            "psnr_gain": float(unet2_tissue["psnr"]) - before_tissue_psnr,
            "ssim_gain": float(unet2_tissue["ssim"]) - before_tissue_ssim,
            "all_psnr": float(unet2_all["psnr"]),
            "all_ssim": float(unet2_all["ssim"]),
        },
        {
            "method": "2.5D U-Net",
            "tissue_psnr": float(unet25_tissue["after_psnr"]),
            "tissue_ssim": float(unet25_tissue["after_ssim"]),
            "psnr_gain": float(unet25_tissue["psnr_gain"]),
            "ssim_gain": float(unet25_tissue["ssim_gain"]),
            "all_psnr": float(unet25_all["after_psnr"]),
            "all_ssim": float(unet25_all["after_ssim"]),
        },
        {
            "method": "2.5D U-Net bf64",
            "tissue_psnr": float(unet25_bf64_tissue["after_psnr"]),
            "tissue_ssim": float(unet25_bf64_tissue["after_ssim"]),
            "psnr_gain": float(unet25_bf64_tissue["psnr_gain"]),
            "ssim_gain": float(unet25_bf64_tissue["ssim_gain"]),
            "all_psnr": float(unet25_bf64_all["after_psnr"]),
            "all_ssim": float(unet25_bf64_all["after_ssim"]),
        },
        {
            "method": "2.5D Residual ResNet",
            "tissue_psnr": float(resnet_tissue["after_psnr"]),
            "tissue_ssim": float(resnet_tissue["after_ssim"]),
            "psnr_gain": float(resnet_tissue["psnr_gain"]),
            "ssim_gain": float(resnet_tissue["ssim_gain"]),
            "all_psnr": float(resnet_all["after_psnr"]),
            "all_ssim": float(resnet_all["after_ssim"]),
        },
    ]
    return pd.DataFrame(rows)


def save_results_outputs(results: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    csv_path = TABLE_DIR / "task2_main_results.csv"
    json_path = METADATA_DIR / "task2_main_results.json"
    md_path = TABLE_DIR / "task2_main_results.md"

    rounded = results.copy()
    for col in ["tissue_psnr", "tissue_ssim", "psnr_gain", "ssim_gain", "all_psnr", "all_ssim"]:
        rounded[col] = rounded[col].round(4)

    rounded.to_csv(csv_path, index=False)
    rounded.to_json(json_path, orient="records", indent=2)

    md_lines = [
        "| Method | Tissue PSNR | Tissue SSIM | PSNR Gain | SSIM Gain |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in rounded.iterrows():
        gain_psnr = "-" if pd.isna(row["psnr_gain"]) else f"+{row['psnr_gain']:.2f}"
        gain_ssim = "-" if pd.isna(row["ssim_gain"]) else f"+{row['ssim_gain']:.3f}"
        md_lines.append(
            f"| {row['method']} | {row['tissue_psnr']:.2f} | {row['tissue_ssim']:.3f} | "
            f"{gain_psnr} | {gain_ssim} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    outputs.extend([csv_path, json_path, md_path])
    return outputs


def plot_main_results(results: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), dpi=180)
    colors = ["#8795a7", "#235789", "#1f7a8c", "#52a3b0", "#4f7f45"]

    labels = results["method"].tolist()
    x = np.arange(len(labels))

    axes[0].bar(x, results["tissue_psnr"], color=colors, width=0.68)
    axes[0].set_title("Tissue-slice PSNR")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_ylim(24, 41)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_xticks(x, labels, rotation=16, ha="right")

    axes[1].bar(x, results["tissue_ssim"], color=colors, width=0.68)
    axes[1].set_title("Tissue-slice SSIM")
    axes[1].set_ylabel("SSIM")
    axes[1].set_ylim(0.68, 1.0)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(x, labels, rotation=16, ha="right")

    for ax, col in zip(axes, ["tissue_psnr", "tissue_ssim"]):
        for i, value in enumerate(results[col]):
            txt = f"{value:.2f}" if col == "tissue_psnr" else f"{value:.3f}"
            ax.text(i, value + (0.35 if col == "tissue_psnr" else 0.006), txt, ha="center", fontsize=8)

    fig.suptitle("Task 2 main results under vertical-line R=5 undersampling", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIGURE_DIR / "main_results_psnr_ssim.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_gain_chart(results: pd.DataFrame) -> Path:
    model_rows = results.iloc[1:].copy()
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    colors = ["#235789", "#1f7a8c", "#52a3b0", "#4f7f45"]
    ax.barh(model_rows["method"], model_rows["psnr_gain"], color=colors)
    ax.set_xlabel("PSNR gain over aliased input (dB)")
    ax.set_title("Residual ResNet gives the largest tissue-slice gain")
    ax.grid(axis="x", alpha=0.25)
    for i, value in enumerate(model_rows["psnr_gain"]):
        ax.text(value + 0.15, i, f"+{value:.2f} dB", va="center", fontsize=9)
    fig.tight_layout()
    out = FIGURE_DIR / "psnr_gain_over_input.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def load_extended_nonzero_summary() -> pd.DataFrame | None:
    path = TABLE_DIR / "task2_nonzero_extended_metrics_summary.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def plot_extended_key_metrics() -> list[Path]:
    summary = load_extended_nonzero_summary()
    if summary is None:
        return []

    outputs: list[Path] = []
    labels = summary["model"].tolist()
    colors = ["#8795a7", "#235789", "#1f7a8c", "#52a3b0", "#4f7f45"]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), dpi=180)
    specs = [
        ("psnr_mean", "PSNR mean (dB)", (26, 41), "{:.2f}"),
        ("ssim_mean", "SSIM mean", (0.68, 1.0), "{:.3f}"),
        ("mae_mean", "MAE mean", (0, max(summary["mae_mean"]) * 1.25), "{:.4f}"),
        ("rmse_mean", "RMSE mean", (0, max(summary["rmse_mean"]) * 1.25), "{:.4f}"),
    ]
    for ax, (col, title, ylim, fmt) in zip(axes.ravel(), specs):
        ax.bar(x, summary[col], color=colors, width=0.68)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.set_xticks(x, labels, rotation=14, ha="right")
        ax.grid(axis="y", alpha=0.25)
        offset = (ylim[1] - ylim[0]) * 0.015
        for idx, value in enumerate(summary[col]):
            ax.text(idx, value + offset, fmt.format(value), ha="center", fontsize=8)
    fig.suptitle("Nonzero tissue-slice metrics, unified evaluation policy", y=1.01, fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIGURE_DIR / "task2_nonzero_key_metrics.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    outputs.append(out)

    table_cols = [
        "model",
        "n_nonzero_slices",
        "psnr_mean",
        "psnr_median",
        "psnr_variance",
        "ssim_mean",
        "ssim_median",
        "ssim_variance",
        "mae_mean",
        "mae_median",
        "rmse_mean",
        "rmse_median",
    ]
    table_df = summary[table_cols].copy()
    rename = {
        "model": "Model",
        "n_nonzero_slices": "n",
        "psnr_mean": "PSNR mean",
        "psnr_median": "PSNR med",
        "psnr_variance": "PSNR var",
        "ssim_mean": "SSIM mean",
        "ssim_median": "SSIM med",
        "ssim_variance": "SSIM var",
        "mae_mean": "MAE mean",
        "mae_median": "MAE med",
        "rmse_mean": "RMSE mean",
        "rmse_median": "RMSE med",
    }
    table_df = table_df.rename(columns=rename)
    for col in ["PSNR mean", "PSNR med", "PSNR var"]:
        table_df[col] = table_df[col].map(lambda v: f"{v:.2f}")
    for col in ["SSIM mean", "SSIM med"]:
        table_df[col] = table_df[col].map(lambda v: f"{v:.3f}")
    table_df["SSIM var"] = table_df["SSIM var"].map(lambda v: f"{v:.6f}")
    for col in ["MAE mean", "MAE med", "RMSE mean", "RMSE med"]:
        table_df[col] = table_df[col].map(lambda v: f"{v:.5f}")

    fig, ax = plt.subplots(figsize=(13.5, 3.8), dpi=180)
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1.0, 1.55)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d8e0ea")
        if r == 0:
            cell.set_facecolor("#e9f0f7")
            cell.set_text_props(weight="bold", color="#18334f")
        elif r == len(table_df):
            cell.set_facecolor("#eaf6f1")
            cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f7fafc")
    fig.suptitle("Extended nonzero/tissue-slice statistics required for PSNR, SSIM, MAE and RMSE", y=0.98, fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = FIGURE_DIR / "task2_extended_metrics_summary_table.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    outputs.append(out)
    return outputs


def load_2d_unet_tissue_metrics() -> tuple[np.ndarray, np.ndarray]:
    path = EXPERIMENTS["2D U-Net"]["eval_dir"] / "psnr_ssim_raw.csv"
    df = pd.read_csv(path)
    tissue = df[df["selected_by_grouping"].astype(bool)]
    return tissue["PSNR_dB"].to_numpy(dtype=np.float32), tissue["SSIM"].to_numpy(dtype=np.float32)


def load_model_tissue_metrics(exp_dir: Path, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(exp_dir / "per_slice_metrics.csv")
    tissue = df[df["is_nonblank"].astype(bool)]
    return tissue[f"{prefix}_psnr"].to_numpy(dtype=np.float32), tissue[f"{prefix}_ssim"].to_numpy(dtype=np.float32)


def plot_consistent_metric_distributions() -> Path:
    input_psnr, input_ssim = load_model_tissue_metrics(EXPERIMENTS["2.5D Residual ResNet"]["dir"], "before")
    unet2_psnr, unet2_ssim = load_2d_unet_tissue_metrics()
    unet25_psnr, unet25_ssim = load_model_tissue_metrics(EXPERIMENTS["2.5D U-Net"]["dir"], "after")
    unet25_bf64_psnr, unet25_bf64_ssim = load_model_tissue_metrics(EXPERIMENTS["2.5D U-Net bf64"]["dir"], "after")
    resnet_psnr, resnet_ssim = load_model_tissue_metrics(EXPERIMENTS["2.5D Residual ResNet"]["dir"], "after")

    labels = ["Input", "2D U-Net", "2.5D U-Net", "2.5D U-Net bf64", "2.5D ResNet"]
    psnr_data = [input_psnr, unet2_psnr, unet25_psnr, unet25_bf64_psnr, resnet_psnr]
    ssim_data = [input_ssim, unet2_ssim, unet25_ssim, unet25_bf64_ssim, resnet_ssim]
    colors = ["#8795a7", "#235789", "#1f7a8c", "#52a3b0", "#4f7f45"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), dpi=180)
    for ax, data, ylabel, title, ylim in [
        (axes[0], psnr_data, "PSNR (dB)", "Tissue-slice PSNR distribution", (24, 58)),
        (axes[1], ssim_data, "SSIM", "Tissue-slice SSIM distribution", (0.68, 1.005)),
    ]:
        parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("#263445")
            body.set_alpha(0.35)
        parts["cmedians"].set_color("#111827")
        parts["cmedians"].set_linewidth(2.0)
        bp = ax.boxplot(
            data,
            widths=0.20,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111827", "linewidth": 1.5},
            boxprops={"facecolor": "white", "edgecolor": "#263445", "linewidth": 1.0},
            whiskerprops={"color": "#263445", "linewidth": 1.0},
            capprops={"color": "#263445", "linewidth": 1.0},
        )
        ax.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=16, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
        for i, values in enumerate(data, start=1):
            ax.text(
                i,
                np.percentile(values, 98),
                f"mean {np.mean(values):.2f}" if ylabel.startswith("PSNR") else f"mean {np.mean(values):.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#344254",
            )

    fig.suptitle("Consistent test distribution: non-blank tissue slices only", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIGURE_DIR / "consistent_tissue_metric_distributions.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _extract_old_metric(text: str, section_label: str, metric_label: str) -> float | None:
    section_idx = text.find(section_label)
    if section_idx < 0:
        return None
    tail = text[section_idx : section_idx + 700]
    pattern = rf"{re.escape(metric_label)}\s*:?\s*([0-9.]+)"
    match = re.search(pattern, tail)
    return float(match.group(1)) if match else None


def make_pointwise_exploration_chart() -> list[Path]:
    """Summarize point-wise mask experiments as a separate exploratory regime."""
    outputs: list[Path] = []
    old_task2 = OLD_REPO / "outputs" / "task2"
    before_path = old_task2 / "2d_unet_baseline_eval_target_nonzero" / "metrics_before_recon.txt"
    unet2_path = old_task2 / "2d_unet_baseline_eval_target_nonzero" / "metrics.txt"
    unet25_path = old_task2 / "final_25d_unet_ctx3_nonzero_full" / "metrics.txt"

    if not before_path.exists() or not unet2_path.exists() or not unet25_path.exists():
        return outputs

    before_text = before_path.read_text(encoding="utf-8", errors="ignore")
    unet2_text = unet2_path.read_text(encoding="utf-8", errors="ignore")
    unet25_text = unet25_path.read_text(encoding="utf-8", errors="ignore")

    rows = [
        {
            "regime": "Point-wise exploratory",
            "method": "Input",
            "psnr": _extract_old_metric(before_text, "Non-blank slices", "PSNR mean"),
            "ssim": _extract_old_metric(before_text, "Non-blank slices", "SSIM mean"),
        },
        {
            "regime": "Point-wise exploratory",
            "method": "2D U-Net",
            "psnr": _extract_old_metric(unet2_text, "Non-blank slices", "PSNR Mean"),
            "ssim": _extract_old_metric(unet2_text, "Non-blank slices", "SSIM Mean"),
        },
        {
            "regime": "Point-wise exploratory",
            "method": "2.5D U-Net",
            "psnr": _extract_old_metric(unet25_text, "[Non-blank / tissue slices]", "After PSNR"),
            "ssim": _extract_old_metric(unet25_text, "[Non-blank / tissue slices]", "After SSIM"),
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "pointwise_exploration_results.csv", index=False)
    outputs.append(TABLE_DIR / "pointwise_exploration_results.csv")

    labels = df["method"].tolist()
    x = np.arange(len(labels))
    colors = ["#8795a7", "#235789", "#1f7a8c"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), dpi=180)
    axes[0].bar(x, df["psnr"], color=colors)
    axes[0].set_title("Point-wise mask tissue PSNR")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_ylim(26, 43)
    axes[0].set_xticks(x, labels, rotation=12, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, df["ssim"], color=colors)
    axes[1].set_title("Point-wise mask tissue SSIM")
    axes[1].set_ylabel("SSIM")
    axes[1].set_ylim(0.30, 1.02)
    axes[1].set_xticks(x, labels, rotation=12, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    for ax, col in zip(axes, ["psnr", "ssim"]):
        for idx, value in enumerate(df[col]):
            txt = f"{value:.2f}" if col == "psnr" else f"{value:.3f}"
            ax.text(idx, value + (0.35 if col == "psnr" else 0.015), txt, ha="center", fontsize=8)
    fig.suptitle("Exploratory point-wise mask regime: useful contrast, not the final physical setting", y=1.02, fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = FIGURE_DIR / "pointwise_exploration_results.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    outputs.append(out)
    return outputs


def make_contact_sheet(image_paths: list[Path], labels: list[str], out: Path, thumb_width: int = 540) -> Path:
    thumbs: list[Image.Image] = []
    font = ImageFont.load_default()
    label_h = 34
    pad = 18
    for path, label in zip(image_paths, labels):
        img = Image.open(path).convert("RGB")
        scale = thumb_width / img.width
        thumb = img.resize((thumb_width, max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_width, thumb.height + label_h), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 10), label, fill=(20, 35, 55), font=font)
        canvas.paste(thumb, (0, label_h))
        thumbs.append(canvas)

    height = max(t.height for t in thumbs)
    width = sum(t.width for t in thumbs) + pad * (len(thumbs) - 1)
    sheet = Image.new("RGB", (width, height), "white")
    x = 0
    for t in thumbs:
        sheet.paste(t, (x, 0))
        x += t.width + pad
    sheet.save(out)
    return out


def copy_existing_figures() -> list[Path]:
    outputs: list[Path] = []
    copies = [
        (
            EXPERIMENTS["2D U-Net"]["train_dir"] / "loss_curve.png",
            FIGURE_DIR / "loss_curve_2d_unet.png",
        ),
        (
            EXPERIMENTS["2.5D U-Net"]["dir"] / "loss_curve.png",
            FIGURE_DIR / "loss_curve_25d_unet.png",
        ),
        (
            EXPERIMENTS["2.5D U-Net bf64"]["dir"] / "loss_curve.png",
            FIGURE_DIR / "loss_curve_25d_unet_bf64.png",
        ),
        (
            EXPERIMENTS["2.5D Residual ResNet"]["dir"] / "loss_curve.png",
            FIGURE_DIR / "loss_curve_25d_resnet.png",
        ),
        (
            EXPERIMENTS["2D U-Net"]["eval_dir"] / "reconstruction_samples.png",
            FIGURE_DIR / "reconstruction_samples_2d_unet.png",
        ),
        (
            EXPERIMENTS["2.5D U-Net"]["dir"] / "reconstruction_samples.png",
            FIGURE_DIR / "reconstruction_samples_25d_unet.png",
        ),
        (
            EXPERIMENTS["2.5D U-Net bf64"]["dir"] / "reconstruction_samples.png",
            FIGURE_DIR / "reconstruction_samples_25d_unet_bf64.png",
        ),
        (
            EXPERIMENTS["2.5D Residual ResNet"]["dir"] / "reconstruction_samples.png",
            FIGURE_DIR / "reconstruction_samples_25d_resnet.png",
        ),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copy2(src, dst)
            outputs.append(dst)

    loss_sheet = make_contact_sheet(
        [dst for _, dst in copies[:4] if dst.exists()],
        ["2D U-Net", "2.5D U-Net bf32", "2.5D U-Net bf64", "2.5D Residual ResNet"],
        FIGURE_DIR / "loss_curves_contact_sheet.png",
        thumb_width=330,
    )
    outputs.append(loss_sheet)
    return outputs


def plot_25d_training_curves() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=180)
    for label, color in [
        ("2.5D U-Net", "#235789"),
        ("2.5D U-Net bf64", "#52a3b0"),
        ("2.5D Residual ResNet", "#1f7a8c"),
    ]:
        hist_path = EXPERIMENTS[label]["dir"] / "history.csv"
        if not hist_path.exists():
            continue
        hist = pd.read_csv(hist_path)
        axes[0].plot(hist["epoch"], hist["train_loss"], label=label, color=color, linewidth=2)
        axes[1].plot(hist["epoch"], hist["val_loss"], label=label, color=color, linewidth=2)

    axes[0].set_title("Training loss")
    axes[1].set_title("Validation loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("2.5D model convergence", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIGURE_DIR / "combined_25d_training_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_25d_capacity_ablation() -> list[Path]:
    outputs: list[Path] = []
    bf32_dir = EXPERIMENTS["2.5D U-Net"]["dir"]
    bf64_dir = EXPERIMENTS["2.5D U-Net bf64"]["dir"]
    if not (bf32_dir / "before_after_comparison.csv").exists() or not (bf64_dir / "before_after_comparison.csv").exists():
        return outputs

    bf32_tissue = read_before_after_csv(bf32_dir / "before_after_comparison.csv", "nonblank_slices")
    bf64_tissue = read_before_after_csv(bf64_dir / "before_after_comparison.csv", "nonblank_slices")
    bf32_all = read_before_after_csv(bf32_dir / "before_after_comparison.csv", "all_slices")
    bf64_all = read_before_after_csv(bf64_dir / "before_after_comparison.csv", "all_slices")

    df = pd.DataFrame(
        [
            {
                "model": "2.5D U-Net bf32",
                "tissue_psnr": float(bf32_tissue["after_psnr"]),
                "tissue_ssim": float(bf32_tissue["after_ssim"]),
                "all_psnr": float(bf32_all["after_psnr"]),
                "all_ssim": float(bf32_all["after_ssim"]),
                "val_loss": float(load_json(bf32_dir / "training_summary.json")["best_loss"]),
            },
            {
                "model": "2.5D U-Net bf64",
                "tissue_psnr": float(bf64_tissue["after_psnr"]),
                "tissue_ssim": float(bf64_tissue["after_ssim"]),
                "all_psnr": float(bf64_all["after_psnr"]),
                "all_ssim": float(bf64_all["after_ssim"]),
                "val_loss": float(load_json(bf64_dir / "training_summary.json")["best_loss"]),
            },
        ]
    )
    csv_path = TABLE_DIR / "25d_unet_capacity_ablation.csv"
    df.to_csv(csv_path, index=False)
    outputs.append(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), dpi=180)
    colors = ["#1f7a8c", "#52a3b0"]
    x = np.arange(len(df))

    axes[0].bar(x, df["tissue_psnr"], color=colors)
    axes[0].set_title("Tissue PSNR")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_ylim(34.5, 36.3)

    axes[1].bar(x, df["tissue_ssim"], color=colors)
    axes[1].set_title("Tissue SSIM")
    axes[1].set_ylabel("SSIM")
    axes[1].set_ylim(0.974, 0.981)

    axes[2].bar(x, df["val_loss"], color=colors)
    axes[2].set_title("Best validation MSE")
    axes[2].set_ylabel("MSE")
    axes[2].set_ylim(0.00028, 0.00034)

    for ax, col in zip(axes, ["tissue_psnr", "tissue_ssim", "val_loss"]):
        ax.set_xticks(x, df["model"], rotation=12, ha="right")
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(df[col]):
            if col == "tissue_psnr":
                txt = f"{value:.2f}"
                offset = 0.04
            elif col == "tissue_ssim":
                txt = f"{value:.4f}"
                offset = 0.00015
            else:
                txt = f"{value:.6f}"
                offset = 0.0000015
            ax.text(idx, value + offset, txt, ha="center", fontsize=8)

    fig.suptitle("2.5D U-Net capacity ablation: widening helps, but only modestly", y=1.02, fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = FIGURE_DIR / "25d_unet_capacity_ablation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    outputs.append(out)

    fig, ax = plt.subplots(figsize=(8.3, 4.4), dpi=180)
    width = 0.35
    ax.bar(x - width / 2, df["tissue_psnr"], width=width, label="Tissue slices", color="#52a3b0")
    ax.bar(x + width / 2, df["all_psnr"], width=width, label="All slices", color="#b8892e")
    ax.set_xticks(x, df["model"], rotation=10, ha="right")
    ax.set_ylabel("PSNR (dB)")
    ax.set_ylim(34, 41)
    ax.set_title("All-slice PSNR can move differently because of blank/background slices")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for idx, row in df.iterrows():
        ax.text(idx - width / 2, row["tissue_psnr"] + 0.12, f"{row['tissue_psnr']:.2f}", ha="center", fontsize=8)
        ax.text(idx + width / 2, row["all_psnr"] + 0.12, f"{row['all_psnr']:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    out = FIGURE_DIR / "25d_unet_bf64_tissue_vs_all_psnr.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    outputs.append(out)
    return outputs


def ifft2c(kspace2d: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(np.fft.ifftshift(kspace2d))


def generate_vertical_line_mask(
    shape: tuple[int, int],
    acceleration: float = 5.0,
    center_fraction: float = 0.10,
    sigma: float = 0.28,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    height, width = shape
    target_columns = int(round(width / acceleration))
    mask = np.zeros((height, width), dtype=bool)

    center_columns = max(4, int(round(width * center_fraction)))
    col_start = (width - center_columns) // 2
    col_end = col_start + center_columns
    mask[:, col_start:col_end] = True

    already_sampled = int(mask[0].sum())
    if already_sampled > target_columns:
        raise ValueError("Center columns exceed sampling budget.")

    rng = np.random.default_rng(seed)
    col_coords = np.arange(width, dtype=np.float32)
    col_coords = (col_coords - (width - 1) / 2.0) / (width / 2.0)
    density = np.exp(-(col_coords**2) / (2.0 * sigma**2)) + 0.01
    density[col_start:col_end] = 0.0

    remaining = target_columns - already_sampled
    if remaining > 0:
        chosen_columns = rng.choice(width, size=remaining, replace=False, p=density / density.sum())
        mask[:, chosen_columns] = True

    achieved = (height * width) / float(mask.sum())
    return mask.astype(np.float32), achieved


def normalize_display(img: np.ndarray, scale: float | None = None) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    if scale is None:
        nz = img[img > 0]
        scale = float(np.percentile(nz, 99)) if nz.size else float(np.max(img) or 1.0)
    return np.clip(img / max(scale, 1e-8), 0.0, 1.0)


def read_nii_array(path: Path) -> np.ndarray:
    """Read a simple uncompressed or gzipped NIfTI-1 image as a NumPy array.

    This is deliberately small: it supports the datatype set used by common
    BraTS NIfTI files and avoids requiring nibabel for presentation assets.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        header = f.read(348)
        if len(header) != 348:
            raise ValueError(f"{path} is too small to be a NIfTI-1 file.")

        sizeof_hdr_le = struct.unpack("<i", header[:4])[0]
        sizeof_hdr_be = struct.unpack(">i", header[:4])[0]
        if sizeof_hdr_le == 348:
            endian = "<"
        elif sizeof_hdr_be == 348:
            endian = ">"
        else:
            raise ValueError(f"{path} has invalid NIfTI header size.")

        dims = struct.unpack(endian + "8h", header[40:56])
        ndim = int(dims[0])
        shape = tuple(int(v) for v in dims[1 : ndim + 1])
        datatype = struct.unpack(endian + "h", header[70:72])[0]
        vox_offset = int(round(struct.unpack(endian + "f", header[108:112])[0]))
        scl_slope = struct.unpack(endian + "f", header[112:116])[0]
        scl_inter = struct.unpack(endian + "f", header[116:120])[0]

        dtype_map = {
            2: np.uint8,
            4: np.int16,
            8: np.int32,
            16: np.float32,
            64: np.float64,
            256: np.int8,
            512: np.uint16,
            768: np.uint32,
            1024: np.int64,
            1280: np.uint64,
        }
        if datatype not in dtype_map:
            raise ValueError(f"Unsupported NIfTI datatype {datatype} in {path}.")

        dtype = np.dtype(dtype_map[datatype]).newbyteorder(endian)
        f.seek(vox_offset)
        count = int(np.prod(shape))
        data = np.frombuffer(f.read(count * dtype.itemsize), dtype=dtype, count=count)
        if data.size != count:
            raise ValueError(f"Could not read expected data payload from {path}.")

    arr = data.reshape(shape, order="F").astype(np.float32, copy=False)
    if scl_slope not in (0.0, 1.0) or scl_inter != 0.0:
        slope = 1.0 if scl_slope == 0.0 else float(scl_slope)
        arr = arr * slope + float(scl_inter)
    return arr


def find_t2w_nii(root: Path, patient_id: str) -> Path:
    candidates = [
        root / patient_id / f"{patient_id}-t2w.nii",
        root / patient_id / f"{patient_id}-t2w.nii.gz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No T2w NIfTI found for {patient_id} under {root}.")


def robust_scale(volume: np.ndarray, percentile: float = 99.0) -> float:
    nonzero = volume[volume > 0]
    if nonzero.size == 0:
        return 1.0
    scale = float(np.percentile(nonzero, percentile))
    return scale if scale > 0 else 1.0


def normalize_shared(slice_2d: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        return np.zeros_like(slice_2d, dtype=np.float32)
    return np.clip(np.asarray(slice_2d, dtype=np.float32) / scale, 0.0, 1.0)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


if nn is not None:

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int, dilation: int = 1) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
                nn.GroupNorm(_group_count(channels), channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(_group_count(channels), channels),
            )
            self.activation = nn.SiLU(inplace=True)

        def forward(self, x):
            return self.activation(x + self.block(x))


    class ResNet25D(nn.Module):
        def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 1,
            base_channels: int = 64,
            num_blocks: int = 12,
            dilations: tuple[int, ...] = (1, 2, 4, 1),
        ) -> None:
            super().__init__()
            self.center_index = in_channels // 2
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(_group_count(base_channels), base_channels),
                nn.SiLU(inplace=True),
            )
            self.body = nn.Sequential(
                *[
                    ResidualBlock(base_channels, dilation=dilations[idx % len(dilations)])
                    for idx in range(num_blocks)
                ]
            )
            self.head = nn.Sequential(
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(_group_count(base_channels), base_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(base_channels, out_channels, kernel_size=1),
            )

        def forward(self, x):
            center = x[:, self.center_index : self.center_index + 1]
            residual = self.head(self.body(self.stem(x)))
            return center + residual


def select_kspace_example() -> tuple[Path, int, np.ndarray]:
    split_path = EXPERIMENTS["2.5D Residual ResNet"]["dir"] / "split_patients.json"
    candidates: list[str] = []
    if split_path.exists():
        splits = load_json(split_path)
        candidates = list(splits.get("test", []))

    kspace_root = GIT_DIR / "kspace_t2w_slicewise_fft"
    for patient_id in candidates:
        path = kspace_root / patient_id / f"{patient_id}-t2w_kspace_complex.npz"
        if path.exists():
            data = np.load(path)["kspace"]
            break
    else:
        path = next(kspace_root.glob("*/*_kspace_complex.npz"))
        data = np.load(path)["kspace"]

    scores = []
    for idx in range(data.shape[2]):
        full = np.abs(ifft2c(data[:, :, idx]))
        scores.append(np.percentile(full, 99))
    slice_idx = int(np.argmax(scores))
    return path, slice_idx, data


def make_kspace_assets() -> list[Path]:
    outputs: list[Path] = []
    kspace_path, slice_idx, kspace = select_kspace_example()
    k2d = kspace[:, :, slice_idx]
    mask, achieved = generate_vertical_line_mask(k2d.shape)
    full = np.abs(ifft2c(k2d)).astype(np.float32)
    aliased = np.abs(ifft2c(k2d * mask)).astype(np.float32)
    err = np.abs(full - aliased)
    scale = float(np.percentile(full[full > 0], 99)) if np.any(full > 0) else float(full.max() or 1.0)

    mask_path = FIGURE_DIR / "vertical_line_mask_r5_seed42.png"
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=180)
    ax.imshow(mask.T, cmap="gray", origin="lower", aspect="auto")
    ax.set_title(f"Vertical-line mask, R~{achieved:.2f}")
    ax.set_axis_off()
    fig.tight_layout(pad=0.2)
    fig.savefig(mask_path, bbox_inches="tight")
    plt.close(fig)
    outputs.append(mask_path)

    pipeline_path = FIGURE_DIR / "line_mask_pipeline_example.png"
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.8), dpi=180)
    panels = [
        (mask.T, "Vertical-line mask", "gray", None),
        (normalize_display(full, scale), "Fully sampled target", "gray", None),
        (normalize_display(aliased, scale), "Aliased input", "gray", None),
        (normalize_display(err), "Absolute aliasing error", "magma", None),
    ]
    for ax, (img, title, cmap, _) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, origin="lower")
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    fig.suptitle(f"{kspace_path.parent.name}, slice {slice_idx:03d}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(pipeline_path, bbox_inches="tight")
    plt.close(fig)
    outputs.append(pipeline_path)

    kspace_demo = FIGURE_DIR / "kspace_masking_demo.png"
    log_full = np.log1p(np.abs(k2d))
    log_masked = np.log1p(np.abs(k2d * mask))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), dpi=180)
    for ax, img, title in zip(
        axes,
        [normalize_display(log_full), mask.T, normalize_display(log_masked)],
        ["Full k-space log magnitude", "Line mask", "Masked k-space log magnitude"],
    ):
        ax.imshow(img.T if title != "Line mask" else img, cmap="gray", origin="lower")
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(kspace_demo, bbox_inches="tight")
    plt.close(fig)
    outputs.append(kspace_demo)

    meta = {
        "kspace_file": str(kspace_path.relative_to(GIT_DIR)),
        "patient_id": kspace_path.parent.name,
        "slice_index": slice_idx,
        "mask_acceleration": achieved,
        "mask_sampled_fraction": float(mask.mean()),
        "display_scale_target_p99": scale,
    }
    meta_path = METADATA_DIR / "selected_kspace_example.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    outputs.append(meta_path)
    return outputs


def make_mask_comparison() -> Path | None:
    old_mask = OLD_REPO / "outputs" / "task1" / "submission_r5_deliverables" / "01_mask" / "variable_density_mask_r5.png"
    line_mask = FIGURE_DIR / "vertical_line_mask_r5_seed42.png"
    if not old_mask.exists() or not line_mask.exists():
        return None
    out = FIGURE_DIR / "pointwise_vs_line_mask_comparison.png"
    make_contact_sheet(
        [old_mask, line_mask],
        ["Preliminary point-wise mask", "Final vertical-line mask"],
        out,
        thumb_width=480,
    )
    return out


def make_real_nii_pair_example() -> list[Path]:
    outputs: list[Path] = []
    meta_path = METADATA_DIR / "selected_kspace_example.json"
    if meta_path.exists():
        meta = load_json(meta_path)
        patient_id = meta["patient_id"]
        slice_idx = int(meta["slice_index"])
    else:
        kspace_path, slice_idx, _ = select_kspace_example()
        patient_id = kspace_path.parent.name

    try:
        us_vol = read_nii_array(find_t2w_nii(LINE_UNDERSAMPLED_ROOT, patient_id))
        fs_vol = read_nii_array(find_t2w_nii(ARCHIVE_ROOT, patient_id))
    except Exception as exc:
        print(f"Skipping NIfTI pair example: {exc}")
        return outputs

    num_slices = min(us_vol.shape[2], fs_vol.shape[2])
    slice_idx = min(max(slice_idx, 0), num_slices - 1)
    scale = robust_scale(fs_vol[:, :, :num_slices])
    us = normalize_shared(us_vol[:, :, slice_idx], scale)
    fs = normalize_shared(fs_vol[:, :, slice_idx], scale)
    err = np.abs(us - fs)

    out = FIGURE_DIR / "real_line_input_target_pair.png"
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.7), dpi=180)
    panels = [
        (us, "Aliased input from line-mask data", "gray"),
        (fs, "Fully sampled target", "gray"),
        (err, "Absolute input error", "magma"),
    ]
    for ax, (img, title, cmap) in zip(axes, panels):
        ax.imshow(img.T, cmap=cmap, origin="lower", vmin=0.0, vmax=1.0 if cmap == "gray" else None)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    fig.suptitle(f"Real NIfTI pair: {patient_id}, slice {slice_idx:03d}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    outputs.append(out)
    return outputs


def load_resnet_model():
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is not available.")
    ckpt_path = EXPERIMENTS["2.5D Residual ResNet"]["dir"] / "best_resnet25d.pth"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    context_slices = int(config.get("context_slices", 3))
    base_channels = int(checkpoint.get("base_channels", config.get("base_channels", 64)))
    num_blocks = int(checkpoint.get("num_blocks", config.get("num_blocks", 12)))
    dilations = tuple(checkpoint.get("dilations", config.get("dilations_tuple", (1, 2, 4, 1))))
    model = ResNet25D(
        in_channels=context_slices,
        base_channels=base_channels,
        num_blocks=num_blocks,
        dilations=dilations,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, context_slices


def make_resnet_worst_case_visuals(num_cases: int = 5) -> list[Path]:
    outputs: list[Path] = []
    metrics_path = EXPERIMENTS["2.5D Residual ResNet"]["dir"] / "per_slice_metrics.csv"
    if not metrics_path.exists():
        return outputs
    if torch is None:
        print("Skipping ResNet worst-case visuals: PyTorch is unavailable.")
        return outputs

    df = pd.read_csv(metrics_path)
    tissue = df[df["is_nonblank"].astype(bool)].copy()
    tissue["psnr_gain"] = tissue["after_psnr"] - tissue["before_psnr"]
    tissue["ssim_gain"] = tissue["after_ssim"] - tissue["before_ssim"]
    worst = tissue.sort_values("after_psnr").head(num_cases).reset_index(drop=True)

    model, context_slices = load_resnet_model()
    half = context_slices // 2
    cache: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    samples = []

    for _, row in worst.iterrows():
        patient_id = str(row["patient_id"])
        slice_idx = int(row["slice_index"])
        if patient_id not in cache:
            us_vol = read_nii_array(find_t2w_nii(LINE_UNDERSAMPLED_ROOT, patient_id))
            fs_vol = read_nii_array(find_t2w_nii(ARCHIVE_ROOT, patient_id))
            num_slices = min(us_vol.shape[2], fs_vol.shape[2])
            scale = robust_scale(fs_vol[:, :, :num_slices])
            cache[patient_id] = (us_vol[:, :, :num_slices], fs_vol[:, :, :num_slices], scale)

        us_vol, fs_vol, scale = cache[patient_id]
        z = min(max(slice_idx, 0), us_vol.shape[2] - 1)
        input_slices = []
        for offset in range(-half, half + 1):
            zz = min(max(z + offset, 0), us_vol.shape[2] - 1)
            input_slices.append(normalize_shared(us_vol[:, :, zz], scale))
        target = normalize_shared(fs_vol[:, :, z], scale)
        center = input_slices[half]

        with torch.no_grad():
            tensor = torch.from_numpy(np.stack(input_slices, axis=0)[None].astype(np.float32))
            recon = model(tensor).cpu().numpy()[0, 0]
        recon = np.clip(recon, 0.0, 1.0)
        error = np.abs(recon - target)
        samples.append((row, center, recon, target, error))

    out = FIGURE_DIR / "resnet_worst5_tissue_visuals.png"
    fig, axes = plt.subplots(len(samples), 4, figsize=(12.2, 3.0 * len(samples)), dpi=180)
    if len(samples) == 1:
        axes = np.expand_dims(axes, axis=0)
    for r, (row, center, recon, target, error) in enumerate(samples):
        panels = [
            (center, f"Aliased input\nBefore {row['before_psnr']:.2f} dB", "gray", 1.0),
            (recon, f"ResNet recon\nAfter {row['after_psnr']:.2f} dB", "gray", 1.0),
            (target, "Ground truth", "gray", 1.0),
            (error, "Absolute error", "magma", None),
        ]
        for c, (img, title, cmap, vmax) in enumerate(panels):
            axes[r, c].imshow(img.T, cmap=cmap, origin="lower", vmin=0.0, vmax=vmax)
            axes[r, c].set_title(title, fontsize=9)
            axes[r, c].set_axis_off()
        axes[r, 0].set_ylabel(f"{row['patient_id']}\nz={int(row['slice_index'])}", fontsize=8)
    fig.suptitle("Worst 5 tissue slices for 2.5D residual ResNet by after-PSNR", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    outputs.append(out)

    table_out = TABLE_DIR / "resnet_worst5_tissue_visual_cases.csv"
    worst.to_csv(table_out, index=False)
    outputs.append(table_out)
    return outputs


def save_worst_case_tables() -> list[Path]:
    outputs: list[Path] = []
    for label, exp in [
        ("25d_unet", EXPERIMENTS["2.5D U-Net"]),
        ("25d_unet_bf64", EXPERIMENTS["2.5D U-Net bf64"]),
        ("25d_resnet", EXPERIMENTS["2.5D Residual ResNet"]),
    ]:
        path = exp["dir"] / "per_slice_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        tissue = df[df["is_nonblank"].astype(bool)].copy()
        tissue["psnr_gain"] = tissue["after_psnr"] - tissue["before_psnr"]
        tissue["ssim_gain"] = tissue["after_ssim"] - tissue["before_ssim"]
        worst = tissue.sort_values("after_psnr").head(10)
        best_gain = tissue.sort_values("psnr_gain", ascending=False).head(10)
        worst_path = TABLE_DIR / f"{label}_worst_tissue_slices.csv"
        best_path = TABLE_DIR / f"{label}_best_gain_tissue_slices.csv"
        worst.to_csv(worst_path, index=False)
        best_gain.to_csv(best_path, index=False)
        outputs.extend([worst_path, best_path])
    return outputs


def save_experiment_notes(results: pd.DataFrame) -> Path:
    lines = [
        "# Task 2 presentation asset notes",
        "",
        "Main interpretation:",
        "",
        "- Use vertical-line R=5 results as final Task 2 evidence.",
        "- Treat point-wise mask experiments as a separate artifact-regime comparison.",
        "- Do not claim that 2.5D context is universally better than 2D.",
        "- The wider 2.5D U-Net bf64 is a capacity ablation: it improves over bf32, but only modestly.",
        "- Strongest supported claim: residual artifact correction performs best under Cartesian line undersampling.",
        "",
        "Main tissue-slice results:",
        "",
    ]
    for _, row in results.iterrows():
        if row["method"].startswith("Input"):
            lines.append(f"- {row['method']}: PSNR {row['tissue_psnr']:.2f}, SSIM {row['tissue_ssim']:.3f}")
        else:
            lines.append(
                f"- {row['method']}: PSNR {row['tissue_psnr']:.2f}, SSIM {row['tissue_ssim']:.3f}, "
                f"gain +{row['psnr_gain']:.2f} dB / +{row['ssim_gain']:.3f}"
            )
    lines += [
        "",
        "Useful caveats:",
        "",
        "- 2D U-Net and 2.5D U-Net are not fully capacity-matched.",
        "- The bf64 ablation reduces the capacity concern but does not overturn the residual-correction conclusion.",
        "- Current Task 2 models are image-domain models without explicit k-space data consistency.",
        "- Tissue-slice metrics are primary because background slices can bias all-slice scores.",
    ]
    out = NOTE_DIR / "presentation_notes.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def save_manifest(outputs: list[Path]) -> Path:
    manifest = []
    for path in sorted(set(outputs)):
        if path is None:
            continue
        if path.exists():
            manifest.append(
                {
                    "file": str(path.relative_to(PRE_DIR)),
                    "bytes": path.stat().st_size,
                }
            )
    out = METADATA_DIR / "asset_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def main() -> None:
    ensure_dirs()
    outputs: list[Path] = []

    results = build_results_table()
    outputs += save_results_outputs(results)
    outputs.append(plot_main_results(results))
    outputs.append(plot_gain_chart(results))
    outputs += plot_extended_key_metrics()
    outputs.append(plot_consistent_metric_distributions())
    outputs += copy_existing_figures()
    outputs.append(plot_25d_training_curves())
    outputs += plot_25d_capacity_ablation()
    outputs += make_kspace_assets()
    outputs += make_real_nii_pair_example()
    outputs += make_pointwise_exploration_chart()

    mask_comparison = make_mask_comparison()
    if mask_comparison is not None:
        outputs.append(mask_comparison)

    outputs += save_worst_case_tables()
    outputs += make_resnet_worst_case_visuals(num_cases=5)
    outputs.append(save_experiment_notes(results))
    manifest = save_manifest(outputs)

    print(f"Generated {len(outputs)} asset files.")
    print(f"Asset directory: {ASSET_DIR}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
