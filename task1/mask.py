from __future__ import annotations

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib.pyplot as plt

LOCAL_INPUT_ROOT = PROJECT_ROOT.parent / "archive"
LOCAL_KSPACE_ROOT = PROJECT_ROOT / "outputs" / "task1" / "kspace_t2w_slicewise_fft"
LOCAL_PREVIEW_DIR = PROJECT_ROOT / "outputs" / "task1" / "undersampling_preview"
LOCAL_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task1" / "undersampled_raw_data_t2w_r5"
LEGACY_INPUT_ROOT = Path("raw_data")
LEGACY_KSPACE_ROOT = Path("kspace_t2w_slicewise_fft")
LEGACY_PREVIEW_DIR = Path("outputs") / "undersampling_preview"
LEGACY_OUTPUT_ROOT = Path("undersampled_raw_data_t2w_r5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview or batch-generate random variable-density undersampled T2w reconstructions."
    )
    parser.add_argument(
        "--path-profile",
        choices=["local", "legacy"],
        default="local",
        help="local uses the current repo-relative data layout; legacy keeps the original relative defaults.",
    )
    parser.add_argument(
        "--mode",
        choices=["preview", "batch"],
        default="preview",
        help="preview: show one random slice; batch: reconstruct all T2w volumes with random per-slice masks.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Root directory containing BraTS case folders and original T2w NIfTI files.",
    )
    parser.add_argument(
        "--kspace-root",
        type=Path,
        default=None,
        help="Root directory containing precomputed slice-wise centered complex k-space .npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory used for preview figures.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Batch output root. Folder structure mirrors raw_data, but each case folder stores only undersampled T2w.",
    )
    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help="Optional case directory name, for example BraTS-GLI-00000-000.",
    )
    parser.add_argument(
        "--case-index",
        type=int,
        default=None,
        help="Optional sorted case index. Ignored when --case-id is set.",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="Optional slice index for preview mode. When omitted, a non-empty slice is chosen at random.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of batch cases to process.",
    )
    parser.add_argument(
        "--acceleration",
        type=float,
        default=5.0,
        help="Target acceleration factor R. R=5 means about 20%% k-space samples are kept.",
    )
    parser.add_argument(
        "--center-fraction",
        type=float,
        default=0.10,
        help="Side-length fraction of the fully-sampled low-frequency square in k-space.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.28,
        help="Spread of the Gaussian-shaped variable-density sampling profile.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible mask and slice selection.",
    )
    parser.add_argument(
        "--save-batch-preview",
        action="store_true",
        help="In batch mode, also save one preview figure for the first processed case.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    if args.path_profile == "legacy":
        input_root = LEGACY_INPUT_ROOT
        kspace_root = LEGACY_KSPACE_ROOT
        output_dir = LEGACY_PREVIEW_DIR
        output_root = LEGACY_OUTPUT_ROOT
    else:
        input_root = LOCAL_INPUT_ROOT
        kspace_root = LOCAL_KSPACE_ROOT
        output_dir = LOCAL_PREVIEW_DIR
        output_root = LOCAL_OUTPUT_ROOT

    args.input_root = args.input_root if args.input_root is not None else input_root
    args.kspace_root = args.kspace_root if args.kspace_root is not None else kspace_root
    args.output_dir = args.output_dir if args.output_dir is not None else output_dir
    args.output_root = args.output_root if args.output_root is not None else output_root


def find_t2w_files(input_root: Path) -> list[Path]:
    return sorted(input_root.rglob("*-t2w.nii")) + sorted(input_root.rglob("*-t2w.nii.gz"))


def find_kspace_files(kspace_root: Path) -> list[Path]:
    return sorted(kspace_root.rglob("*_kspace_complex.npz"))


def load_volume(path: Path) -> np.ndarray:
    image = nib.load(str(path))
    return np.asarray(image.get_fdata(), dtype=np.float32)


def ifft2c(kspace2d: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(np.fft.ifftshift(kspace2d))


def choose_case(t2w_files: list[Path], case_id: str | None, case_index: int | None, rng: np.random.Generator) -> Path:
    if not t2w_files:
        raise FileNotFoundError("No T2w files were found under the input root.")

    if case_id is not None:
        for path in t2w_files:
            if path.parent.name == case_id:
                return path
        raise FileNotFoundError(f"Case {case_id!r} was not found under the input root.")

    if case_index is not None:
        bounded_index = min(max(case_index, 0), len(t2w_files) - 1)
        return t2w_files[bounded_index]

    return t2w_files[int(rng.integers(0, len(t2w_files)))]


def choose_slice(volume: np.ndarray, requested_index: int | None, rng: np.random.Generator) -> int:
    if requested_index is not None:
        return min(max(requested_index, 0), volume.shape[2] - 1)

    nonzero_fraction = np.mean(volume > 0, axis=(0, 1))
    candidate_slices = np.flatnonzero(nonzero_fraction > 0.08)
    if candidate_slices.size == 0:
        return volume.shape[2] // 2
    return int(rng.choice(candidate_slices))


def generate_variable_density_mask(
    shape: tuple[int, int],
    acceleration: float,
    center_fraction: float,
    sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    if acceleration <= 1.0:
        raise ValueError("Acceleration must be greater than 1.0 for undersampling.")
    if not (0.0 < center_fraction < 1.0):
        raise ValueError("center_fraction must be between 0 and 1.")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")

    height, width = shape
    total_points = height * width
    target_samples = int(round(total_points / acceleration))

    mask = np.zeros((height, width), dtype=bool)

    center_h = max(4, int(round(height * center_fraction)))
    center_w = max(4, int(round(width * center_fraction)))
    row_start = (height - center_h) // 2
    row_end = row_start + center_h
    col_start = (width - center_w) // 2
    col_end = col_start + center_w
    mask[row_start:row_end, col_start:col_end] = True

    already_sampled = int(mask.sum())
    if already_sampled > target_samples:
        raise ValueError(
            "The fully sampled center is larger than the allowed sample budget. "
            "Reduce center_fraction or acceleration."
        )

    yy, xx = np.mgrid[0:height, 0:width]
    yy = (yy - (height - 1) / 2.0) / (height / 2.0)
    xx = (xx - (width - 1) / 2.0) / (width / 2.0)
    radius = np.sqrt(xx**2 + yy**2)

    density = np.exp(-(radius**2) / (2.0 * sigma**2)) + 0.01
    density[mask] = 0.0

    remaining = target_samples - already_sampled
    if remaining > 0:
        flat_density = density.reshape(-1)
        chosen = rng.choice(
            flat_density.size,
            size=remaining,
            replace=False,
            p=flat_density / flat_density.sum(),
        )
        mask.reshape(-1)[chosen] = True

    achieved_acceleration = total_points / float(mask.sum())
    return mask.astype(np.float32), achieved_acceleration


def save_mask_preview(mask: np.ndarray, output_path: Path, acceleration: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(mask.T, cmap="gray", origin="lower")
    ax.set_title(f"Random Variable-Density Mask (R~{acceleration:.2f})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_comparison_figure(
    full_image: np.ndarray,
    mask: np.ndarray,
    aliased_image: np.ndarray,
    output_path: Path,
    case_name: str,
    slice_index: int,
    acceleration: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vmax = float(np.percentile(full_image, 99.5))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    axes[0].imshow(mask.T, cmap="gray", origin="lower")
    axes[0].set_title(f"Sampling Mask\nR~{acceleration:.2f}")
    axes[0].axis("off")

    axes[1].imshow(full_image.T, cmap="gray", origin="lower", vmin=0.0, vmax=vmax)
    axes[1].set_title(f"Fully Sampled\n{case_name} slice {slice_index}")
    axes[1].axis("off")

    axes[2].imshow(aliased_image.T, cmap="gray", origin="lower", vmin=0.0, vmax=vmax)
    axes[2].set_title("Aliased / Undersampled")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def case_id_from_kspace_path(kspace_path: Path) -> str:
    return kspace_path.parent.name


def find_original_t2w_path(input_root: Path, case_id: str) -> Path:
    case_dir = input_root / case_id
    nii_path = case_dir / f"{case_id}-t2w.nii"
    nii_gz_path = case_dir / f"{case_id}-t2w.nii.gz"
    if nii_path.exists():
        return nii_path
    if nii_gz_path.exists():
        return nii_gz_path
    raise FileNotFoundError(f"Could not find original T2w NIfTI for case {case_id}.")


def build_output_t2w_path(output_root: Path, original_t2w_path: Path) -> Path:
    case_id = original_t2w_path.parent.name
    output_case_dir = output_root / case_id
    output_case_dir.mkdir(parents=True, exist_ok=True)
    return output_case_dir / original_t2w_path.name


def reconstruct_volume_from_kspace(
    kspace_volume: np.ndarray,
    acceleration: float,
    center_fraction: float,
    sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    reconstructed = np.zeros(kspace_volume.shape, dtype=np.float32)
    sampled_fraction_sum = 0.0
    first_mask: np.ndarray | None = None

    for slice_idx in range(kspace_volume.shape[2]):
        mask, _ = generate_variable_density_mask(
            shape=kspace_volume.shape[:2],
            acceleration=acceleration,
            center_fraction=center_fraction,
            sigma=sigma,
            rng=rng,
        )
        undersampled_kspace = kspace_volume[:, :, slice_idx] * mask
        reconstructed[:, :, slice_idx] = np.abs(ifft2c(undersampled_kspace)).astype(np.float32)
        sampled_fraction_sum += float(mask.mean())
        if first_mask is None:
            first_mask = mask

    average_sampled_fraction = sampled_fraction_sum / float(kspace_volume.shape[2])
    if first_mask is None:
        raise ValueError("Empty k-space volume encountered.")
    achieved_acceleration = 1.0 / average_sampled_fraction
    return reconstructed, first_mask, achieved_acceleration


def save_reconstructed_t2w(volume: np.ndarray, reference_path: Path, output_path: Path) -> None:
    reference_img = nib.load(str(reference_path))
    reconstructed_img = nib.Nifti1Image(volume.astype(np.float32), reference_img.affine, reference_img.header.copy())
    nib.save(reconstructed_img, str(output_path))


def run_preview(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)

    t2w_files = find_t2w_files(args.input_root)
    selected_path = choose_case(t2w_files, args.case_id, args.case_index, rng)
    volume = load_volume(selected_path)
    slice_index = choose_slice(volume, args.slice_index, rng)
    full_slice = volume[:, :, slice_index]
    full_kspace = np.fft.fftshift(np.fft.fft2(full_slice))

    mask, achieved_acceleration = generate_variable_density_mask(
        shape=full_slice.shape,
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        sigma=args.sigma,
        rng=rng,
    )

    undersampled_kspace = full_kspace * mask
    aliased_slice = np.abs(ifft2c(undersampled_kspace)).astype(np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = args.output_dir / "variable_density_mask_r5_preview.png"
    comparison_path = args.output_dir / f"{selected_path.stem}_slice_{slice_index:03d}_undersampling_preview.png"
    mask_array_path = args.output_dir / "variable_density_mask_r5.npy"

    save_mask_preview(mask, mask_path, achieved_acceleration)
    save_comparison_figure(
        full_image=full_slice,
        mask=mask,
        aliased_image=aliased_slice,
        output_path=comparison_path,
        case_name=selected_path.parent.name,
        slice_index=slice_index,
        acceleration=achieved_acceleration,
    )
    np.save(mask_array_path, mask)

    print(f"Selected T2w case: {selected_path}")
    print(f"Volume shape: {volume.shape}")
    print(f"Selected slice index: {slice_index}")
    print(f"2D slice shape: {full_slice.shape}")
    print(f"Target acceleration: {args.acceleration:.2f}")
    print(f"Achieved acceleration: {achieved_acceleration:.4f}")
    print(f"Mask sampled fraction: {mask.mean():.4f}")
    print(f"Mask preview: {mask_path.resolve()}")
    print(f"Comparison preview: {comparison_path.resolve()}")
    print(f"Mask array: {mask_array_path.resolve()}")


def run_batch(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    kspace_files = find_kspace_files(args.kspace_root)
    if not kspace_files:
        raise FileNotFoundError(f"No k-space .npz files found under {args.kspace_root}.")

    if args.case_id is not None:
        kspace_files = [path for path in kspace_files if case_id_from_kspace_path(path) == args.case_id]
    elif args.case_index is not None:
        bounded_index = min(max(args.case_index, 0), len(kspace_files) - 1)
        kspace_files = [kspace_files[bounded_index]]

    if args.limit is not None:
        kspace_files = kspace_files[: max(args.limit, 0)]

    if not kspace_files:
        raise ValueError("No k-space files selected for batch processing.")

    args.output_root.mkdir(parents=True, exist_ok=True)

    first_preview_saved = False
    processed = 0

    for kspace_path in kspace_files:
        case_id = case_id_from_kspace_path(kspace_path)
        original_t2w_path = find_original_t2w_path(args.input_root, case_id)
        output_t2w_path = build_output_t2w_path(args.output_root, original_t2w_path)

        with np.load(kspace_path) as data:
            kspace_volume = data["kspace"]

        reconstructed_volume, first_mask, achieved_acceleration = reconstruct_volume_from_kspace(
            kspace_volume=kspace_volume,
            acceleration=args.acceleration,
            center_fraction=args.center_fraction,
            sigma=args.sigma,
            rng=rng,
        )
        save_reconstructed_t2w(reconstructed_volume, original_t2w_path, output_t2w_path)

        print(
            f"[{processed + 1}/{len(kspace_files)}] {case_id}: "
            f"shape={reconstructed_volume.shape}, R~{achieved_acceleration:.4f}, "
            f"saved={output_t2w_path}"
        )

        if args.save_batch_preview and not first_preview_saved:
            reference_volume = load_volume(original_t2w_path)
            preview_slice_index = choose_slice(reference_volume, None, rng)
            preview_output = args.output_dir / f"{case_id}_slice_{preview_slice_index:03d}_batch_preview.png"
            save_comparison_figure(
                full_image=reference_volume[:, :, preview_slice_index],
                mask=first_mask,
                aliased_image=reconstructed_volume[:, :, preview_slice_index],
                output_path=preview_output,
                case_name=case_id,
                slice_index=preview_slice_index,
                acceleration=achieved_acceleration,
            )
            first_preview_saved = True

        processed += 1

    print(f"Processed cases: {processed}")
    print(f"Batch output root: {args.output_root.resolve()}")


def main() -> None:
    args = parse_args()
    resolve_paths(args)
    print(f"Path profile: {args.path_profile}")
    print(f"Input root  : {args.input_root.resolve()}")
    print(f"K-space root: {args.kspace_root.resolve()}")
    print(f"Preview dir : {args.output_dir.resolve()}")
    print(f"Output root : {args.output_root.resolve()}")
    if args.mode == "preview":
        run_preview(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
