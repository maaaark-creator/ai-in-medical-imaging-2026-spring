import argparse
import os
import random
import warnings
from pathlib import Path
import numpy as np
import nibabel as nib
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_UNDERSAMPLED_DIR = Path(os.path.expanduser(
    "~/Desktop/project1_without_rawdata/undersampled_raw_data_t2w_r5"
))
LEGACY_FULLY_SAMPLED_DIR = Path(os.path.expanduser("~/Downloads/dataset/archive"))
LEGACY_OUTPUT_DIR = Path(os.path.expanduser("~/Desktop/project1_without_rawdata/task2_final_deliverables"))
LOCAL_FULLY_SAMPLED_DIR = PROJECT_ROOT.parent / "archive"
LOCAL_UNDERSAMPLED_DIR = PROJECT_ROOT / "outputs" / "task1" / "undersampled_raw_data_t2w_r5"
LOCAL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "task2" / "unet_baseline"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
RANDOM_SEED = 42
BACKGROUND_PSNR_THRESH = 56.0


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
    return parser.parse_args()


def resolve_paths(args):
    if args.path_profile == "legacy":
        undersampled_dir = LEGACY_UNDERSAMPLED_DIR
        fully_sampled_dir = LEGACY_FULLY_SAMPLED_DIR
        output_dir = LEGACY_OUTPUT_DIR
    else:
        undersampled_dir = LOCAL_UNDERSAMPLED_DIR
        fully_sampled_dir = LOCAL_FULLY_SAMPLED_DIR
        output_dir = LOCAL_OUTPUT_DIR

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


def norm_slice(s: np.ndarray) -> np.ndarray:
    mn, mx = s.min(), s.max()
    if mx > mn:
        return (s - mn) / (mx - mn)
    return np.zeros_like(s)


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

    test_patients = get_test_patients(undersampled_dir, fully_sampled_dir, args.seed)
    print(f"Total test patients: {len(test_patients)}")

    psnr_all, ssim_all = [], []

    for pid in tqdm(test_patients, desc="Processing"):
        us_path = find_t2w_path(undersampled_dir, pid)
        fs_path = find_t2w_path(fully_sampled_dir, pid)

        if not (os.path.exists(us_path) and os.path.exists(fs_path)):
            print(f"Missing file for {pid}, skipping")
            continue

        us_vol = nib.load(us_path).get_fdata().astype(np.float32)
        fs_vol = nib.load(fs_path).get_fdata().astype(np.float32)

        num_slices = min(us_vol.shape[2], fs_vol.shape[2])
        for s in range(num_slices):
            inp = norm_slice(us_vol[:, :, s])
            tar = norm_slice(fs_vol[:, :, s])

            psnr_val, ssim_val = compute_psnr_ssim(inp, tar)
            psnr_all.append(psnr_val)
            ssim_all.append(ssim_val)

    psnr_all = np.array(psnr_all)
    ssim_all = np.array(ssim_all)

    avg_psnr_full = np.mean(psnr_all)
    avg_ssim_full = np.mean(ssim_all)

    tissue_mask = psnr_all <= BACKGROUND_PSNR_THRESH
    psnr_tissue = psnr_all[tissue_mask]
    ssim_tissue = ssim_all[tissue_mask]
    avg_psnr_tissue = np.mean(psnr_tissue) if len(psnr_tissue) > 0 else 0.0
    avg_ssim_tissue = np.mean(ssim_tissue) if len(ssim_tissue) > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"Total slices: {len(psnr_all)}")
    print(
        f"All slices        -> PSNR: {avg_psnr_full:.4f} dB, SSIM: {avg_ssim_full:.4f}"
    )
    print(
        f"Tissue slices ({np.sum(tissue_mask)}) -> PSNR: {avg_psnr_tissue:.4f} dB, SSIM: {avg_ssim_tissue:.4f}"
    )

    out_path = output_dir / "metrics_before_recon.txt"
    with open(out_path, "w") as f:
        f.write("=== Before Reconstruction (Input vs. Ground Truth) ===\n")
        f.write(f"Total slices: {len(psnr_all)}\n\n")
        f.write("All slices:\n")
        f.write(f"  PSNR mean: {avg_psnr_full:.6f} dB\n")
        f.write(f"  SSIM mean: {avg_ssim_full:.6f}\n\n")
        f.write(f"Tissue slices (PSNR <= {BACKGROUND_PSNR_THRESH} dB):\n")
        f.write(f"  Count    : {len(psnr_tissue)}\n")
        f.write(f"  PSNR mean: {avg_psnr_tissue:.6f} dB\n")
        f.write(f"  SSIM mean: {avg_ssim_tissue:.6f}\n\n")
        f.write(
            "Per-slice PSNR (all):\n" + ", ".join(f"{v:.6f}" for v in psnr_all) + "\n"
        )
        f.write(
            "Per-slice SSIM (all):\n" + ", ".join(f"{v:.6f}" for v in ssim_all) + "\n"
        )

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
