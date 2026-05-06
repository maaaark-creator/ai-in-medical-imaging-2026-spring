import os
import random
import warnings
import numpy as np
import nibabel as nib
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

UNDERSAMPLED_DIR = os.path.expanduser(
    "~/Desktop/project1_without_rawdata/undersampled_raw_data_t2w_r5"
)
FULLY_SAMPLED_DIR = os.path.expanduser("~/Downloads/dataset/archive")
OUTPUT_DIR = os.path.expanduser(
    "~/Desktop/project1_without_rawdata/task2_final_deliverables"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
RANDOM_SEED = 42
BACKGROUND_PSNR_THRESH = 56.0


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


def get_test_patients():
    us_pats = sorted(os.listdir(UNDERSAMPLED_DIR))
    fs_pats = sorted(os.listdir(FULLY_SAMPLED_DIR))
    common = sorted(list(set(us_pats) & set(fs_pats)))
    random.seed(RANDOM_SEED)
    random.shuffle(common)
    n_train = int(len(common) * TRAIN_RATIO)
    n_val = int(len(common) * VAL_RATIO)
    return common[n_train + n_val :]


def main():
    test_patients = get_test_patients()
    print(f"Total test patients: {len(test_patients)}")

    psnr_all, ssim_all = [], []

    for pid in tqdm(test_patients, desc="Processing"):
        us_path = os.path.join(UNDERSAMPLED_DIR, pid, f"{pid}-t2w.nii")
        fs_path = os.path.join(FULLY_SAMPLED_DIR, pid, f"{pid}-t2w.nii")

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

    out_path = os.path.join(OUTPUT_DIR, "metrics_before_recon.txt")
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
