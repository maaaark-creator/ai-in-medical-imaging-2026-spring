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
LOCAL_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task1" / "kspace_t2w_slicewise_fft"
LOCAL_PREVIEW_DIR = PROJECT_ROOT / "outputs" / "task1" / "kspace_previews"
LEGACY_INPUT_ROOT = Path("raw_data")
LEGACY_OUTPUT_ROOT = Path("kspace_t2w_slicewise_fft")
LEGACY_PREVIEW_DIR = Path("outputs") / "kspace_previews"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert all T2w volumes to complex k-space with slice-wise 2D FFT.")
    parser.add_argument(
        "--path-profile",
        choices=["local", "legacy"],
        default="local",
        help="local uses the current repo-relative data layout; legacy keeps the original relative defaults.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Root directory containing BraTS case folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory where slice-wise complex k-space files will be written.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Directory for preview PNG files.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Which converted case to use for preview generation.",
    )
    parser.add_argument(
        "--preview-slice-index",
        type=int,
        default=-1,
        help="Slice index used for preview generation. Negative values select the center slice.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of T2w volumes to convert.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    if args.path_profile == "legacy":
        input_root = LEGACY_INPUT_ROOT
        output_root = LEGACY_OUTPUT_ROOT
        preview_dir = LEGACY_PREVIEW_DIR
    else:
        input_root = LOCAL_INPUT_ROOT
        output_root = LOCAL_OUTPUT_ROOT
        preview_dir = LOCAL_PREVIEW_DIR

    args.input_root = args.input_root if args.input_root is not None else input_root
    args.output_root = args.output_root if args.output_root is not None else output_root
    args.preview_dir = args.preview_dir if args.preview_dir is not None else preview_dir


def find_t2w_files(input_root: Path) -> list[Path]:
    return sorted(input_root.rglob("*-t2w.nii")) + sorted(input_root.rglob("*-t2w.nii.gz"))


def to_complex_kspace_2d(volume: np.ndarray) -> np.ndarray:
    # Match the common MRI teaching setup: 2D FFT on each slice, then center k-space in x/y.
    kspace = np.fft.fft2(volume, axes=(0, 1))
    return np.fft.fftshift(kspace, axes=(0, 1)).astype(np.complex64)


def make_output_path(input_root: Path, output_root: Path, t2w_path: Path) -> Path:
    relative_path = t2w_path.relative_to(input_root)
    stem = relative_path.name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    elif stem.endswith(".nii"):
        stem = stem[:-4]
    output_name = f"{stem}_kspace_complex.npz"
    return output_root / relative_path.parent / output_name


def save_complex_kspace(kspace: np.ndarray, reference_img: nib.Nifti1Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        kspace=kspace,
        affine=reference_img.affine.astype(np.float32),
        shape=np.array(kspace.shape, dtype=np.int32),
        spacing=np.array(reference_img.header.get_zooms()[:3], dtype=np.float32),
    )


def resolve_slice_index(volume: np.ndarray, requested_index: int) -> int:
    if requested_index < 0:
        return volume.shape[2] // 2
    return min(max(requested_index, 0), volume.shape[2] - 1)


def save_preview(original: np.ndarray, kspace: np.ndarray, preview_path: Path, slice_idx: int) -> None:
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    original_slice = original[:, :, slice_idx]
    kspace_slice = np.log1p(np.abs(kspace[:, :, slice_idx]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_slice.T, cmap="gray", origin="lower")
    axes[0].set_title(f"Original T2w Slice {slice_idx}")
    axes[0].axis("off")

    axes[1].imshow(kspace_slice.T, cmap="gray", origin="lower")
    axes[1].set_title(f"K-space Log Magnitude Slice {slice_idx}")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(preview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    resolve_paths(args)
    print(f"Path profile: {args.path_profile}")
    print(f"Input root  : {args.input_root.resolve()}")
    print(f"Output root : {args.output_root.resolve()}")
    print(f"Preview dir : {args.preview_dir.resolve()}")
    t2w_files = find_t2w_files(args.input_root)
    if not t2w_files:
        raise FileNotFoundError(f"No T2w files found under {args.input_root}")
    if args.limit is not None:
        t2w_files = t2w_files[: max(args.limit, 0)]
    if not t2w_files:
        raise ValueError("No T2w files selected after applying --limit.")

    converted: list[tuple[Path, Path]] = []

    for t2w_path in t2w_files:
        image = nib.load(str(t2w_path))
        volume = np.asarray(image.get_fdata(), dtype=np.float32)
        kspace = to_complex_kspace_2d(volume)

        output_path = make_output_path(args.input_root, args.output_root, t2w_path)
        save_complex_kspace(kspace, image, output_path)
        converted.append((t2w_path, output_path))

    sample_index = min(max(args.sample_index, 0), len(converted) - 1)
    sample_input, sample_output = converted[sample_index]
    sample_volume = np.asarray(nib.load(str(sample_input)).get_fdata(), dtype=np.float32)
    sample_kspace = np.load(sample_output)["kspace"]
    preview_slice_index = resolve_slice_index(sample_volume, args.preview_slice_index)

    preview_name = f"{sample_input.stem}_slice_{preview_slice_index:03d}_kspace_preview.png"
    preview_path = args.preview_dir / preview_name
    save_preview(sample_volume, sample_kspace, preview_path, preview_slice_index)

    print(f"Converted {len(converted)} T2w volumes to complex k-space.")
    print("FFT mode: 2D FFT per slice on axes (0, 1), then fftshift on axes (0, 1).")
    print(f"Output root: {args.output_root.resolve()}")
    print("Each .npz file stores the FFT result for every slice in the corresponding T2w volume.")
    print(f"Sample input: {sample_input}")
    print(f"Sample output: {sample_output}")
    print(f"Preview slice index: {preview_slice_index}")
    print(f"Preview image: {preview_path.resolve()}")


if __name__ == "__main__":
    main()
