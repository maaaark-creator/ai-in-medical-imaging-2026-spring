import argparse
import csv
import os
import random
import warnings
from pathlib import Path
import numpy as np
import nibabel as nib
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
from slice_grouping import (
    DEFAULT_BACKGROUND_PSNR_THRESHOLD,
    DEFAULT_SLICE_GROUPING,
    DEFAULT_TARGET_NONZERO_THRESHOLD,
    SLICE_GROUPING_CHOICES,
    build_nonblank_mask,
    default_eval_output_dir,
    grouping_title,
    target_nonzero_fraction,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_UNDERSAMPLED_DIR = Path(os.path.expanduser(
    "~/Desktop/project1_without_rawdata/undersampled_raw_data_t2w_r5"
))
LEGACY_FULLY_SAMPLED_DIR = Path(os.path.expanduser("~/Downloads/dataset/archive"))
LEGACY_OUTPUT_DIR = Path(os.path.expanduser("~/Desktop/project1_without_rawdata/task2_final_deliverables"))
LOCAL_DATA_ROOT = PROJECT_ROOT.parent
LOCAL_FULLY_SAMPLED_DIR = LOCAL_DATA_ROOT / "archive"
LOCAL_UNDERSAMPLED_DIR = LOCAL_DATA_ROOT / "undersampled_raw_data_t2w_vertical_line_r5"
LOCAL_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task2"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
RANDOM_SEED = 42
DEFAULT_NORM_MODE = "target-volume-robust"
DEFAULT_ROBUST_PERCENTILE = 99.0


def expand_path(path: Path) -> Path:
    return Path(os.path.expanduser(str(path))).resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="Compute input-vs-ground-truth PSNR/SSIM before neural reconstruction.")
    parser.add_argument(
        "--path-profile",
        choices=["local", "legacy"],
        default="local",
        help="local uses repo-relative paths; legacy keeps the original cloud-platform paths.",
    )
    parser.add_argument("--undersampled-dir", type=Path, default=None, help="Override undersampled T2w root.")
    parser.add_argument("--fully-sampled-dir", type=Path, default=None, help="Override fully sampled BraTS root.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--norm-mode",
        choices=["separate", "target-volume-robust"],
        default=DEFAULT_NORM_MODE,
        help="separate: per-slice min-max; target-volume-robust: input/target share target-volume pXX scale.",
    )
    parser.add_argument("--robust-percentile", type=float, default=DEFAULT_ROBUST_PERCENTILE)
    parser.add_argument(
        "--slice-grouping",
        choices=SLICE_GROUPING_CHOICES,
        default=DEFAULT_SLICE_GROUPING,
        help="How to define the non-blank/tissue slice subset for reporting.",
    )
    parser.add_argument(
        "--target-nonzero-threshold",
        type=float,
        default=DEFAULT_TARGET_NONZERO_THRESHOLD,
        help="Target-slice nonzero fraction threshold used when --slice-grouping=target_nonzero.",
    )
    parser.add_argument(
        "--background-psnr-threshold",
        type=float,
        default=DEFAULT_BACKGROUND_PSNR_THRESHOLD,
        help="PSNR threshold used when --slice-grouping=psnr_threshold.",
    )
    return parser.parse_args()


def resolve_paths(args):
    if args.path_profile == "legacy":
        undersampled_dir = LEGACY_UNDERSAMPLED_DIR
        fully_sampled_dir = LEGACY_FULLY_SAMPLED_DIR
        output_dir = default_eval_output_dir(LEGACY_OUTPUT_DIR, args.slice_grouping)
    else:
        undersampled_dir = LOCAL_UNDERSAMPLED_DIR
        fully_sampled_dir = LOCAL_FULLY_SAMPLED_DIR
        output_dir = default_eval_output_dir(LOCAL_OUTPUT_ROOT, args.slice_grouping)

    if args.undersampled_dir is not None:
        undersampled_dir = args.undersampled_dir
    if args.fully_sampled_dir is not None:
        fully_sampled_dir = args.fully_sampled_dir
    if args.output_dir is not None:
        output_dir = args.output_dir

    return expand_path(undersampled_dir), expand_path(fully_sampled_dir), expand_path(output_dir)


def find_t2w_path(root, patient_id):
    root = Path(root)
    candidates = [
        root / patient_id / f"{patient_id}-t2w.nii",
        root / patient_id / f"{patient_id}-t2w.nii.gz",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def robust_target_scale(volume: np.ndarray, percentile: float) -> float:
    tissue_values = volume[volume > 0]
    if tissue_values.size == 0:
        return 1.0
    scale = float(np.percentile(tissue_values, percentile))
    if scale <= 0.0:
        scale = float(tissue_values.max())
    return max(scale, 1e-6)


def norm_slice(s: np.ndarray) -> np.ndarray:
    mn, mx = s.min(), s.max()
    if mx > mn:
        return ((s - mn) / (mx - mn)).astype(np.float32)
    return np.zeros_like(s, dtype=np.float32)


def norm_with_scale(s: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0.0:
        return np.zeros_like(s, dtype=np.float32)
    return np.clip(s / scale, 0.0, 1.0).astype(np.float32)


def compute_psnr_ssim(input_slice, target_slice):
    mse = np.mean((input_slice - target_slice) ** 2)
    if mse == 0:
        psnr = 100.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            psnr = peak_signal_noise_ratio(target_slice, input_slice, data_range=1.0)
            if np.isinf(psnr):
                psnr = 100.0

    if np.std(target_slice) < 1e-8 or np.std(input_slice) < 1e-8:
        ssim = 1.0 if np.allclose(input_slice, target_slice) else 0.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ssim = structural_similarity(target_slice, input_slice, data_range=1.0)
    return float(psnr), float(ssim)


def get_test_patients(undersampled_dir, fully_sampled_dir, seed):
    us_pats = sorted(os.listdir(undersampled_dir))
    fs_pats = sorted(os.listdir(fully_sampled_dir))
    common = sorted(list(set(us_pats) & set(fs_pats)))
    random.seed(seed)
    random.shuffle(common)
    n_train = int(len(common) * TRAIN_RATIO)
    n_val = int(len(common) * VAL_RATIO)
    return common[n_train + n_val :]


def main():
    args = parse_args()
    undersampled_dir, fully_sampled_dir, output_dir = resolve_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Path profile       : {args.path_profile}")
    print(f"Undersampled root  : {undersampled_dir}")
    print(f"Fully sampled root : {fully_sampled_dir}")
    print(f"Output dir         : {output_dir}")
    print(f"Slice grouping     : {args.slice_grouping}")
    print(f"Norm mode          : {args.norm_mode}")

    test_patients = get_test_patients(undersampled_dir, fully_sampled_dir, args.seed)
    print(f"Total test patients: {len(test_patients)}")

    psnr_all, ssim_all = [], []
    target_nonzero_all = []

    for pid in tqdm(test_patients, desc="Processing"):
        us_path = find_t2w_path(undersampled_dir, pid)
        fs_path = find_t2w_path(fully_sampled_dir, pid)

        if not (os.path.exists(us_path) and os.path.exists(fs_path)):
            print(f"Missing file for {pid}, skipping")
            continue

        us_vol = nib.load(us_path).get_fdata().astype(np.float32)
        fs_vol = nib.load(fs_path).get_fdata().astype(np.float32)

        num_slices = min(us_vol.shape[2], fs_vol.shape[2])
        norm_scale = (
            robust_target_scale(fs_vol[:, :, :num_slices], args.robust_percentile)
            if args.norm_mode == "target-volume-robust"
            else 1.0
        )
        for s in range(num_slices):
            raw_target = fs_vol[:, :, s]
            if args.norm_mode == "target-volume-robust":
                inp = norm_with_scale(us_vol[:, :, s], norm_scale)
                tar = norm_with_scale(raw_target, norm_scale)
            else:
                inp = norm_slice(us_vol[:, :, s])
                tar = norm_slice(raw_target)

            psnr_val, ssim_val = compute_psnr_ssim(inp, tar)
            psnr_all.append(psnr_val)
            ssim_all.append(ssim_val)
            target_nonzero_all.append(target_nonzero_fraction(raw_target))

    psnr_all = np.array(psnr_all)
    ssim_all = np.array(ssim_all)
    target_nonzero_all = np.array(target_nonzero_all)

    avg_psnr_full = np.mean(psnr_all)
    avg_ssim_full = np.mean(ssim_all)

    nonblank_mask = build_nonblank_mask(
        psnr_all,
        target_nonzero_all,
        args.slice_grouping,
        args.target_nonzero_threshold,
        args.background_psnr_threshold,
    )
    psnr_nonblank = psnr_all[nonblank_mask]
    ssim_nonblank = ssim_all[nonblank_mask]
    avg_psnr_nonblank = np.mean(psnr_nonblank) if len(psnr_nonblank) > 0 else 0.0
    avg_ssim_nonblank = np.mean(ssim_nonblank) if len(ssim_nonblank) > 0 else 0.0
    nonblank_title = grouping_title(
        args.slice_grouping,
        args.target_nonzero_threshold,
        args.background_psnr_threshold,
    )

    print(f"\n{'='*50}")
    print(f"Total slices: {len(psnr_all)}")
    print(
        f"All slices        -> PSNR: {avg_psnr_full:.4f} dB, SSIM: {avg_ssim_full:.4f}"
    )
    print(
        f"{nonblank_title} ({np.sum(nonblank_mask)}) -> "
        f"PSNR: {avg_psnr_nonblank:.4f} dB, SSIM: {avg_ssim_nonblank:.4f}"
    )

    out_path = output_dir / "metrics_before_recon.txt"
    with open(out_path, "w") as f:
        f.write("=== Before Reconstruction (Input vs. Ground Truth) ===\n")
        f.write(f"Total slices: {len(psnr_all)}\n\n")
        f.write("All slices:\n")
        f.write(f"  PSNR mean: {avg_psnr_full:.6f} dB\n")
        f.write(f"  SSIM mean: {avg_ssim_full:.6f}\n\n")
        f.write(f"{nonblank_title}:\n")
        f.write(f"  Count    : {len(psnr_nonblank)}\n")
        if args.slice_grouping == "target_nonzero":
            f.write(
                "  Definition: "
                f"target_nonzero_fraction >= {args.target_nonzero_threshold:g}\n"
            )
        else:
            f.write(
                "  Definition: "
                f"PSNR <= {args.background_psnr_threshold:g} dB\n"
            )
        f.write(f"  PSNR mean: {avg_psnr_nonblank:.6f} dB\n")
        f.write(f"  SSIM mean: {avg_ssim_nonblank:.6f}\n\n")
        f.write(
            "Per-slice PSNR (all):\n" + ", ".join(f"{v:.6f}" for v in psnr_all) + "\n"
        )
        f.write(
            "Per-slice SSIM (all):\n" + ", ".join(f"{v:.6f}" for v in ssim_all) + "\n"
        )
        f.write(
            "Per-slice target_nonzero_fraction (all):\n"
            + ", ".join(f"{v:.6f}" for v in target_nonzero_all)
            + "\n"
        )

    csv_path = output_dir / "before_recon_per_slice_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Index",
                "PSNR_dB",
                "SSIM",
                "target_nonzero_fraction",
                "selected_by_grouping",
            ]
        )
        for idx, (psnr_val, ssim_val, frac, keep) in enumerate(
            zip(psnr_all, ssim_all, target_nonzero_all, nonblank_mask)
        ):
            writer.writerow(
                [idx, f"{psnr_val:.6f}", f"{ssim_val:.6f}", f"{frac:.6f}", int(keep)]
            )

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
