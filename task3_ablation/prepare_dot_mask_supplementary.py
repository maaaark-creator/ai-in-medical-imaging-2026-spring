from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity

from mask import generate_variable_density_mask, ifft2c, load_volume


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export supplementary dot-mask metrics and a compact visual comparison."
    )
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        default=Path("outputs")
        / "submission_vertical_line_deliverables"
        / "04_sample_reference_slices"
        / "sample_reference_manifest.csv",
        help="Manifest containing the 24 reference cases used by the vertical-mask analysis.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "supplementary_dot_mask",
        help="Output directory for dot-mask supplementary results.",
    )
    parser.add_argument("--acceleration", type=float, default=5.0)
    parser.add_argument("--center-fraction", type=float, default=0.10)
    parser.add_argument("--sigma", type=float, default=0.28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-visual-examples",
        type=int,
        default=5,
        help="Number of examples included in the visual montage.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def reconstruct_from_mask(full_slice: np.ndarray, mask: np.ndarray) -> np.ndarray:
    full_kspace = np.fft.fftshift(np.fft.fft2(full_slice))
    return np.abs(ifft2c(full_kspace * mask)).astype(np.float32)


def compute_metrics(full_image: np.ndarray, aliased_image: np.ndarray) -> dict[str, float]:
    diff = full_image.astype(np.float32) - aliased_image.astype(np.float32)
    mse = float(np.mean(diff**2))
    peak = float(full_image.max())
    if peak <= 0:
        peak = 1.0
    psnr = 20.0 * math.log10(peak) - 10.0 * math.log10(mse + 1e-12)

    data_range = float(full_image.max() - full_image.min())
    if data_range <= 0:
        data_range = 1.0
    ssim = float(structural_similarity(full_image, aliased_image, data_range=data_range))
    return {
        "psnr_db": psnr,
        "ssim": ssim,
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(mse)),
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, float | int]:
    psnr = np.array([float(row["psnr_db"]) for row in rows], dtype=np.float64)
    ssim = np.array([float(row["ssim"]) for row in rows], dtype=np.float64)
    return {
        "count": int(len(rows)),
        "psnr_mean_db": float(psnr.mean()),
        "psnr_std_db": float(psnr.std()),
        "psnr_min_db": float(psnr.min()),
        "psnr_max_db": float(psnr.max()),
        "ssim_mean": float(ssim.mean()),
        "ssim_std": float(ssim.std()),
        "ssim_min": float(ssim.min()),
        "ssim_max": float(ssim.max()),
    }


def normalize_vmax(image: np.ndarray) -> float:
    vmax = float(np.percentile(image, 99.5))
    return vmax if vmax > 0 else 1.0


def save_mask(mask: np.ndarray, output_path: Path, achieved_acceleration: float) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(mask.T, cmap="gray", origin="lower")
    ax.set_title(f"Dot-based Variable-density Mask (R~{achieved_acceleration:.2f})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_montage(
    examples: list[tuple[str, int, np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(examples), 2, figsize=(7.8, 3.2 * len(examples)))
    axes = np.atleast_2d(axes)

    for row_idx, (case_id, slice_index, full_image, aliased_image) in enumerate(examples):
        vmax = normalize_vmax(full_image)
        axes[row_idx, 0].imshow(full_image.T, cmap="gray", origin="lower", vmin=0.0, vmax=vmax)
        axes[row_idx, 0].set_title(f"{case_id} | Slice {slice_index} | Fully Sampled", fontsize=10, pad=8)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(aliased_image.T, cmap="gray", origin="lower", vmin=0.0, vmax=vmax)
        axes[row_idx, 1].set_title("Dot-mask Aliased", fontsize=10, pad=8)
        axes[row_idx, 1].axis("off")

    fig.suptitle("Dot-mask Downsampling Examples", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
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


def main() -> None:
    args = parse_args()
    manifest_rows = read_manifest(args.sample_manifest)
    if not manifest_rows:
        raise ValueError("No sample rows were found.")

    first_volume = load_volume(Path(manifest_rows[0]["source_t2w_path"]))
    first_slice = first_volume[:, :, int(manifest_rows[0]["slice_index"])]
    rng = np.random.default_rng(args.seed)
    mask, achieved_acceleration = generate_variable_density_mask(
        shape=first_slice.shape,
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        sigma=args.sigma,
        rng=rng,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "dot_variable_density_mask_r5.npy", mask)
    save_mask(mask, args.output_dir / "dot_variable_density_mask_r5.png", achieved_acceleration)

    metric_rows: list[dict[str, object]] = []
    visual_examples: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    for row in manifest_rows:
        case_id = row["case_id"]
        slice_index = int(row["slice_index"])
        volume = load_volume(Path(row["source_t2w_path"]))
        full_slice = volume[:, :, slice_index]
        aliased_slice = reconstruct_from_mask(full_slice, mask)
        metrics = compute_metrics(full_slice, aliased_slice)
        metric_rows.append(
            {
                "case_id": case_id,
                "slice_index": slice_index,
                **metrics,
            }
        )
        if len(visual_examples) < args.num_visual_examples:
            visual_examples.append((case_id, slice_index, full_slice, aliased_slice))

    summary = summarize(metric_rows)
    payload = {
        "settings": {
            "mask_type": "dot_based_variable_density",
            "target_acceleration": float(args.acceleration),
            "achieved_acceleration": float(achieved_acceleration),
            "center_fraction": float(args.center_fraction),
            "sigma": float(args.sigma),
            "seed": int(args.seed),
            "sample_manifest": str(args.sample_manifest.resolve()),
        },
        "summary": summary,
        "samples": metric_rows,
    }
    (args.output_dir / "dot_mask_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_metrics_csv(args.output_dir / "dot_mask_metrics.csv", metric_rows)
    save_montage(visual_examples, args.output_dir / "dot_mask_examples_montage.png")

    print(f"Saved dot-mask supplementary results to: {args.output_dir.resolve()}")
    print(f"Achieved acceleration: {achieved_acceleration:.4f}")
    print(f"Mean PSNR: {summary['psnr_mean_db']:.4f} dB")
    print(f"Mean SSIM: {summary['ssim_mean']:.4f}")


if __name__ == "__main__":
    main()
