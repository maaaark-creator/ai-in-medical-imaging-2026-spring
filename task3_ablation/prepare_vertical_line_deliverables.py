from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity

from mask import find_t2w_files, generate_vertical_line_mask, ifft2c, load_volume


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a clean set of MRI undersampling deliverables with an R=5 "
            "vertical-line mask, plus standalone PSNR/SSIM metrics."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("raw_data"),
        help="Root directory containing original T2w NIfTI files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "submission_vertical_line_deliverables",
        help="Directory where the organized deliverables will be written.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of example case/slice visualizations to export.",
    )
    parser.add_argument(
        "--num-metric-samples",
        type=int,
        default=24,
        help="Number of representative slices used for standalone PSNR/SSIM analysis.",
    )
    parser.add_argument(
        "--acceleration",
        type=float,
        default=5.0,
        help="Target acceleration factor.",
    )
    parser.add_argument(
        "--center-fraction",
        type=float,
        default=0.10,
        help="Fraction of central k-space columns to keep fully sampled.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.28,
        help="Spread of the variable-density column sampling profile.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the shared undersampling mask.",
    )
    return parser.parse_args()


def choose_example_cases(t2w_files: list[Path], count: int) -> list[Path]:
    if not t2w_files:
        raise FileNotFoundError("No T2w files were found under the input root.")

    clamped_count = max(1, min(count, len(t2w_files)))
    indices = np.linspace(0, len(t2w_files) - 1, clamped_count, dtype=int)
    return [t2w_files[int(index)] for index in indices]


def choose_representative_slice(volume: np.ndarray) -> int:
    nonzero_fraction = np.mean(volume > 0, axis=(0, 1))
    candidate_slices = np.flatnonzero(nonzero_fraction > 0.08)
    if candidate_slices.size == 0:
        return volume.shape[2] // 2

    center_index = volume.shape[2] // 2
    closest = np.argmin(np.abs(candidate_slices - center_index))
    return int(candidate_slices[closest])


def normalize_vmax(image: np.ndarray) -> float:
    vmax = float(np.percentile(image, 99.5))
    return vmax if vmax > 0 else 1.0


def reconstruct_from_mask(full_slice: np.ndarray, mask: np.ndarray) -> np.ndarray:
    full_kspace = np.fft.fftshift(np.fft.fft2(full_slice))
    aliased_slice = np.abs(ifft2c(full_kspace * mask)).astype(np.float32)
    return aliased_slice


def compute_psnr(full_image: np.ndarray, aliased_image: np.ndarray) -> float:
    diff = full_image.astype(np.float32) - aliased_image.astype(np.float32)
    mse = float(np.mean(diff**2))
    peak = float(full_image.max())
    if peak <= 0:
        peak = 1.0
    return 20.0 * math.log10(peak) - 10.0 * math.log10(mse + 1e-12)


def compute_ssim(full_image: np.ndarray, aliased_image: np.ndarray) -> float:
    data_range = float(full_image.max() - full_image.min())
    if data_range <= 0:
        data_range = 1.0
    return float(structural_similarity(full_image, aliased_image, data_range=data_range))


def compute_metrics(full_image: np.ndarray, aliased_image: np.ndarray) -> dict[str, float]:
    diff = full_image.astype(np.float32) - aliased_image.astype(np.float32)
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return {
        "psnr_db": compute_psnr(full_image, aliased_image),
        "ssim": compute_ssim(full_image, aliased_image),
        "mae": mae,
        "rmse": rmse,
    }


def summarize_metric_rows(rows: list[dict[str, object]]) -> dict[str, float | int]:
    psnr_values = np.array([float(row["psnr_db"]) for row in rows], dtype=np.float64)
    ssim_values = np.array([float(row["ssim"]) for row in rows], dtype=np.float64)
    mae_values = np.array([float(row["mae"]) for row in rows], dtype=np.float64)
    rmse_values = np.array([float(row["rmse"]) for row in rows], dtype=np.float64)

    return {
        "count": int(len(rows)),
        "psnr_mean_db": float(psnr_values.mean()),
        "psnr_std_db": float(psnr_values.std()),
        "psnr_min_db": float(psnr_values.min()),
        "psnr_max_db": float(psnr_values.max()),
        "ssim_mean": float(ssim_values.mean()),
        "ssim_std": float(ssim_values.std()),
        "ssim_min": float(ssim_values.min()),
        "ssim_max": float(ssim_values.max()),
        "mae_mean": float(mae_values.mean()),
        "rmse_mean": float(rmse_values.mean()),
    }


def save_mask(mask: np.ndarray, output_path: Path, acceleration: float) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(mask, cmap="gray", origin="lower", aspect="auto")
    ax.set_title(f"Vertical-line Mask (R~{acceleration:.2f})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_comparison(
    mask: np.ndarray,
    full_image: np.ndarray,
    aliased_image: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    vmax = normalize_vmax(full_image)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))

    axes[0].imshow(mask, cmap="gray", origin="lower", aspect="auto")
    axes[0].set_title("Sampling Mask")
    axes[0].axis("off")

    axes[1].imshow(full_image.T, cmap="gray", origin="lower", vmin=0.0, vmax=vmax)
    axes[1].set_title("Fully Sampled")
    axes[1].axis("off")

    axes[2].imshow(aliased_image.T, cmap="gray", origin="lower", vmin=0.0, vmax=vmax)
    axes[2].set_title("Aliased / Undersampled")
    axes[2].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_pair(full_image: np.ndarray, aliased_image: np.ndarray, output_path: Path, title: str) -> None:
    vmax = normalize_vmax(full_image)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6))

    axes[0].imshow(full_image.T, cmap="gray", origin="lower", vmin=0.0, vmax=vmax)
    axes[0].set_title("Fully Sampled")
    axes[0].axis("off")

    axes[1].imshow(aliased_image.T, cmap="gray", origin="lower", vmin=0.0, vmax=vmax)
    axes[1].set_title("Aliased / Undersampled")
    axes[1].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_manifest(output_path: Path, rows: list[dict[str, object]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["case_id", "slice_index", "comparison_figure", "pair_figure"])
        for row in rows:
            writer.writerow(
                [
                    row["case_id"],
                    row["slice_index"],
                    row["comparison_figure"],
                    row["pair_figure"],
                ]
            )


def write_metrics_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["case_id", "slice_index", "psnr_db", "ssim", "mae", "rmse"])
        for row in rows:
            writer.writerow(
                [
                    row["case_id"],
                    row["slice_index"],
                    f"{float(row['psnr_db']):.6f}",
                    f"{float(row['ssim']):.6f}",
                    f"{float(row['mae']):.6f}",
                    f"{float(row['rmse']):.6f}",
                ]
            )


def write_metrics_json(
    output_path: Path,
    settings: dict[str, object],
    summary: dict[str, object],
    rows: list[dict[str, object]],
) -> None:
    payload = {
        "settings": settings,
        "summary": summary,
        "samples": rows,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_readme(
    output_dir: Path,
    num_examples: int,
    num_metric_samples: int,
    acceleration: float,
    achieved_acceleration: float,
) -> None:
    readme_path = output_dir / "README.md"
    readme_text = f"""# MRI Undersampling Deliverables (Vertical-line Mask)

This folder contains organized deliverables for Fourier-domain MRI undersampling with a shared vertical-line mask.

## Settings

- Target acceleration factor: `R={acceleration:.1f}`
- Achieved acceleration of the shared mask: `R~{achieved_acceleration:.4f}`
- Number of exported visualization examples: `{num_examples}`
- Number of standalone metric samples: `{num_metric_samples}`

## Folder layout

- `01_mask/`: the shared vertical-line undersampling mask in PNG and NPY format
- `02_comparisons/`: side-by-side comparison figures showing mask, fully sampled image, and aliased image
- `03_image_pairs/`: fully sampled vs aliased image pairs for the same slices
- `examples_manifest.csv`: case and slice metadata for the exported figures
- `example_metrics.csv` / `example_metrics.json`: PSNR/SSIM/MAE/RMSE for the exported figure examples
- `sample_metrics.csv` / `sample_metrics.json`: PSNR/SSIM/MAE/RMSE for a larger representative slice subset
"""
    readme_path.write_text(readme_text, encoding="utf-8")


def build_metric_row(case_id: str, slice_index: int, full_slice: np.ndarray, aliased_slice: np.ndarray) -> dict[str, object]:
    metrics = compute_metrics(full_slice, aliased_slice)
    return {
        "case_id": case_id,
        "slice_index": int(slice_index),
        **metrics,
    }


def main() -> None:
    args = parse_args()

    t2w_files = find_t2w_files(args.input_root)
    example_cases = choose_example_cases(t2w_files, args.num_examples)
    metric_cases = choose_example_cases(t2w_files, args.num_metric_samples)

    sample_volume = load_volume(example_cases[0])
    sample_slice_index = choose_representative_slice(sample_volume)
    sample_slice = sample_volume[:, :, sample_slice_index]

    rng = np.random.default_rng(args.seed)
    mask, achieved_acceleration = generate_vertical_line_mask(
        shape=sample_slice.shape,
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        sigma=args.sigma,
        rng=rng,
    )

    mask_dir = args.output_dir / "01_mask"
    comparison_dir = args.output_dir / "02_comparisons"
    pair_dir = args.output_dir / "03_image_pairs"
    for directory in (mask_dir, comparison_dir, pair_dir):
        directory.mkdir(parents=True, exist_ok=True)

    np.save(mask_dir / "vertical_line_mask_r5.npy", mask)
    save_mask(mask, mask_dir / "vertical_line_mask_r5.png", achieved_acceleration)

    manifest_rows: list[dict[str, object]] = []
    example_metric_rows: list[dict[str, object]] = []
    for case_path in example_cases:
        volume = load_volume(case_path)
        slice_index = choose_representative_slice(volume)
        full_slice = volume[:, :, slice_index]
        aliased_slice = reconstruct_from_mask(full_slice, mask)

        case_id = case_path.parent.name
        stem = f"{case_id}_slice_{slice_index:03d}"
        comparison_name = f"{stem}_comparison.png"
        pair_name = f"{stem}_pair.png"
        title = f"{case_id} | slice {slice_index} | R~{achieved_acceleration:.2f}"

        save_comparison(mask, full_slice, aliased_slice, comparison_dir / comparison_name, title)
        save_pair(full_slice, aliased_slice, pair_dir / pair_name, title)

        manifest_rows.append(
            {
                "case_id": case_id,
                "slice_index": int(slice_index),
                "comparison_figure": f"02_comparisons/{comparison_name}",
                "pair_figure": f"03_image_pairs/{pair_name}",
            }
        )
        example_metric_rows.append(build_metric_row(case_id, slice_index, full_slice, aliased_slice))

    sample_metric_rows: list[dict[str, object]] = []
    for case_path in metric_cases:
        volume = load_volume(case_path)
        slice_index = choose_representative_slice(volume)
        full_slice = volume[:, :, slice_index]
        aliased_slice = reconstruct_from_mask(full_slice, mask)

        case_id = case_path.parent.name
        sample_metric_rows.append(build_metric_row(case_id, slice_index, full_slice, aliased_slice))

    settings = {
        "mask_type": "vertical_line",
        "target_acceleration": float(args.acceleration),
        "achieved_acceleration": float(achieved_acceleration),
        "center_fraction": float(args.center_fraction),
        "sigma": float(args.sigma),
        "seed": int(args.seed),
    }

    example_summary = summarize_metric_rows(example_metric_rows)
    sample_summary = summarize_metric_rows(sample_metric_rows)

    write_manifest(args.output_dir / "examples_manifest.csv", manifest_rows)
    write_metrics_csv(args.output_dir / "example_metrics.csv", example_metric_rows)
    write_metrics_json(
        args.output_dir / "example_metrics.json",
        settings=settings,
        summary=example_summary,
        rows=example_metric_rows,
    )
    write_metrics_csv(args.output_dir / "sample_metrics.csv", sample_metric_rows)
    write_metrics_json(
        args.output_dir / "sample_metrics.json",
        settings=settings,
        summary=sample_summary,
        rows=sample_metric_rows,
    )

    write_readme(
        output_dir=args.output_dir,
        num_examples=len(example_cases),
        num_metric_samples=len(metric_cases),
        acceleration=args.acceleration,
        achieved_acceleration=achieved_acceleration,
    )

    print(f"Saved deliverables to: {args.output_dir.resolve()}")
    print(f"Examples exported: {len(example_cases)}")
    print(f"Metric samples exported: {len(metric_cases)}")
    print(f"Achieved acceleration: {achieved_acceleration:.4f}")
    print(f"Mask: {(mask_dir / 'vertical_line_mask_r5.png').resolve()}")
    print(f"Comparisons: {comparison_dir.resolve()}")
    print(f"Pairs: {pair_dir.resolve()}")


if __name__ == "__main__":
    main()
