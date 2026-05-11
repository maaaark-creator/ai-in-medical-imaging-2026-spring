from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

from mask import find_t2w_files, generate_mask, ifft2c, load_volume, mask_display_name, mask_file_stem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib.pyplot as plt

LOCAL_INPUT_ROOT = PROJECT_ROOT.parent / "archive"
LOCAL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "task1" / "submission_r5_deliverables"
LEGACY_INPUT_ROOT = Path("raw_data")
LEGACY_OUTPUT_DIR = Path("outputs") / "submission_r5_deliverables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a clean set of MRI undersampling deliverables with R=5 variable-density masking."
    )
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
        help="Root directory containing original T2w NIfTI files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the organized deliverables will be written.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of example case/slice visualizations to export.",
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
        help="Side-length fraction of the fully sampled k-space center square.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.28,
        help="Spread of the variable-density sampling profile.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the shared undersampling mask.",
    )
    parser.add_argument(
        "--mask-type",
        choices=["variable_density_2d", "vertical_line"],
        default="vertical_line",
        help="Sampling pattern. vertical_line keeps complete k-space columns for Cartesian line undersampling.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    if args.path_profile == "legacy":
        input_root = LEGACY_INPUT_ROOT
        output_dir = LEGACY_OUTPUT_DIR
    else:
        input_root = LOCAL_INPUT_ROOT
        output_dir = LOCAL_OUTPUT_DIR

    args.input_root = args.input_root if args.input_root is not None else input_root
    args.output_dir = args.output_dir if args.output_dir is not None else output_dir


def choose_example_cases(t2w_files: list[Path], num_examples: int) -> list[Path]:
    if not t2w_files:
        raise FileNotFoundError("No T2w files were found under the input root.")

    clamped_examples = max(1, min(num_examples, len(t2w_files)))
    indices = np.linspace(0, len(t2w_files) - 1, clamped_examples, dtype=int)
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


def save_mask(mask: np.ndarray, output_path: Path, acceleration: float, mask_type: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(mask.T, cmap="gray", origin="lower")
    ax.set_title(f"{mask_display_name(mask_type)} (R~{acceleration:.2f})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_comparison(mask: np.ndarray, full_image: np.ndarray, aliased_image: np.ndarray, output_path: Path, title: str) -> None:
    vmax = normalize_vmax(full_image)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))

    axes[0].imshow(mask.T, cmap="gray", origin="lower")
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


def write_readme(
    output_dir: Path,
    num_examples: int,
    acceleration: float,
    achieved_acceleration: float,
    manifest_rel_path: str,
    mask_type: str,
) -> None:
    readme_path = output_dir / "README.md"
    readme_text = f"""# MRI Undersampling Deliverables

This folder contains organized deliverables for Fourier-domain MRI undersampling with a {mask_display_name(mask_type)}.

## Settings

- Target acceleration factor: `R={acceleration:.1f}`
- Achieved acceleration of the shared mask: `R~{achieved_acceleration:.4f}`
- Mask type: `{mask_type}`
- Number of exported examples: `{num_examples}`

## Folder layout

- `01_mask/`: the shared undersampling mask in PNG and NPY format
- `02_comparisons/`: side-by-side comparison figures showing mask, fully sampled image, and aliased image
- `03_image_pairs/`: fully sampled vs aliased image pairs for the same slices
- `{manifest_rel_path}`: case and slice metadata for each exported example
"""
    readme_path.write_text(readme_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    resolve_paths(args)
    print(f"Path profile: {args.path_profile}")
    print(f"Input root  : {args.input_root.resolve()}")
    print(f"Output dir  : {args.output_dir.resolve()}")

    t2w_files = find_t2w_files(args.input_root)
    example_cases = choose_example_cases(t2w_files, args.num_examples)

    sample_volume = load_volume(example_cases[0])
    sample_slice_index = choose_representative_slice(sample_volume)
    sample_slice = sample_volume[:, :, sample_slice_index]

    rng = np.random.default_rng(args.seed)
    mask, achieved_acceleration = generate_mask(
        shape=sample_slice.shape,
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        sigma=args.sigma,
        rng=rng,
        mask_type=args.mask_type,
    )

    mask_dir = args.output_dir / "01_mask"
    comparison_dir = args.output_dir / "02_comparisons"
    pair_dir = args.output_dir / "03_image_pairs"
    for directory in (mask_dir, comparison_dir, pair_dir):
        directory.mkdir(parents=True, exist_ok=True)

    mask_stem = mask_file_stem(args.mask_type, args.acceleration, args.seed)
    np.save(mask_dir / f"{mask_stem}.npy", mask)
    save_mask(mask, mask_dir / f"{mask_stem}.png", achieved_acceleration, args.mask_type)

    manifest_path = args.output_dir / "examples_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["case_id", "slice_index", "comparison_figure", "pair_figure"])

        for case_path in example_cases:
            volume = load_volume(case_path)
            slice_index = choose_representative_slice(volume)
            full_slice = volume[:, :, slice_index]
            full_kspace = np.fft.fftshift(np.fft.fft2(full_slice))
            aliased_slice = np.abs(ifft2c(full_kspace * mask)).astype(np.float32)

            case_id = case_path.parent.name
            stem = f"{case_id}_slice_{slice_index:03d}"
            comparison_name = f"{stem}_comparison.png"
            pair_name = f"{stem}_pair.png"
            title = f"{case_id} | slice {slice_index} | R~{achieved_acceleration:.2f}"

            save_comparison(mask, full_slice, aliased_slice, comparison_dir / comparison_name, title)
            save_pair(full_slice, aliased_slice, pair_dir / pair_name, title)

            writer.writerow([case_id, slice_index, f"02_comparisons/{comparison_name}", f"03_image_pairs/{pair_name}"])

    write_readme(
        output_dir=args.output_dir,
        num_examples=len(example_cases),
        acceleration=args.acceleration,
        achieved_acceleration=achieved_acceleration,
        manifest_rel_path=manifest_path.name,
        mask_type=args.mask_type,
    )

    print(f"Saved deliverables to: {args.output_dir.resolve()}")
    print(f"Examples exported: {len(example_cases)}")
    print(f"Achieved acceleration: {achieved_acceleration:.4f}")
    print(f"Mask type: {args.mask_type}")
    print(f"Mask: {(mask_dir / f'{mask_stem}.png').resolve()}")
    print(f"Comparisons: {comparison_dir.resolve()}")
    print(f"Pairs: {pair_dir.resolve()}")


if __name__ == "__main__":
    main()
