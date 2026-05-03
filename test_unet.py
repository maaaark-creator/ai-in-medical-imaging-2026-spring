import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
import csv

UNDERSAMPLED_DIR = os.path.expanduser("~/Desktop/project1_without_rawdata/undersampled_raw_data_t2w_r5")
FULLY_SAMPLED_DIR = os.path.expanduser("~/Downloads/dataset/archive")
BEST_MODEL_PATH = os.path.expanduser("~/Desktop/project1_without_rawdata/task2_final_deliverables/best_unet.pth")
OUTPUT_DIR = os.path.expanduser("~/Desktop/project1_without_rawdata/task2_final_deliverables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BACKGROUND_PSNR_THRESH = 56.0 
RANDOM_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class BraTSSliceDataset(Dataset):
    def __init__(self, patient_list, undersampled_root, fully_sampled_root):
        self.volumes = []
        self.slice_index = []
        print("Loading volumes (strict pairing)...")
        for pid in tqdm(patient_list, desc="Loading patients"):
            us_path = os.path.join(undersampled_root, pid, f"{pid}-t2w.nii")
            fs_path = os.path.join(fully_sampled_root, pid, f"{pid}-t2w.nii")
            if not os.path.exists(us_path) or not os.path.exists(fs_path):
                continue
            try:
                us_vol = nib.load(us_path).get_fdata().astype(np.float32)
                fs_vol = nib.load(fs_path).get_fdata().astype(np.float32)
            except:
                continue
            num_slices = min(us_vol.shape[2], fs_vol.shape[2])
            vol_idx = len(self.volumes)
            self.volumes.append((us_vol, fs_vol))
            for s in range(num_slices):
                self.slice_index.append((vol_idx, s))
        print(f"Loaded {len(self.volumes)} patients, total slices: {len(self.slice_index)}")
    def __len__(self):
        return len(self.slice_index)
    def __getitem__(self, idx):
        vol_idx, slice_z = self.slice_index[idx]
        us_vol, fs_vol = self.volumes[vol_idx]
        img_input = us_vol[:, :, slice_z]
        img_target = fs_vol[:, :, slice_z]
        def norm(s):
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn) if mx > mn else np.zeros_like(s)
        img_input = norm(img_input)
        img_target = norm(img_target)
        return torch.from_numpy(img_input).unsqueeze(0), torch.from_numpy(img_target).unsqueeze(0)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        for feat in features:
            self.downs.append(DoubleConv(in_ch, feat))
            in_ch = feat
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        for feat in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feat*2, feat, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feat*2, feat))
        self.final_conv = nn.Conv2d(features[0], out_ch, 1)
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx//2]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx+1](x)
        return self.final_conv(x)


def compute_psnr_ssim(pred, target):
    pred = pred.squeeze().cpu().numpy()
    target = target.squeeze().cpu().numpy()
    psnr = peak_signal_noise_ratio(target, pred, data_range=1.0)
    ssim = structural_similarity(target, pred, data_range=1.0)
    return float(psnr), float(ssim)


def save_metric_distributions(psnr_all, ssim_all, tissue_mask, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(psnr_all, bins=50, color='steelblue', edgecolor='white')
    axes[0, 0].set_title(f'PSNR (All Slices, n={len(psnr_all)})')
    axes[0, 0].set_xlabel('PSNR (dB)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].axvline(np.mean(psnr_all), color='red', linestyle='--', label=f'Mean={np.mean(psnr_all):.2f}')
    axes[0, 0].legend()
    
    psnr_tissue = psnr_all[tissue_mask]
    axes[0, 1].hist(psnr_tissue, bins=50, color='darkorange', edgecolor='white')
    axes[0, 1].set_title(f'PSNR (Tissue Slices, n={len(psnr_tissue)})')
    axes[0, 1].set_xlabel('PSNR (dB)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].axvline(np.mean(psnr_tissue), color='red', linestyle='--', label=f'Mean={np.mean(psnr_tissue):.2f}')
    axes[0, 1].legend()
    
    axes[1, 0].hist(ssim_all, bins=50, color='steelblue', edgecolor='white')
    axes[1, 0].set_title(f'SSIM (All Slices, n={len(ssim_all)})')
    axes[1, 0].set_xlabel('SSIM')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].axvline(np.mean(ssim_all), color='red', linestyle='--', label=f'Mean={np.mean(ssim_all):.4f}')
    axes[1, 0].legend()
    
    ssim_tissue = ssim_all[tissue_mask]
    axes[1, 1].hist(ssim_tissue, bins=50, color='darkorange', edgecolor='white')
    axes[1, 1].set_title(f'SSIM (Tissue Slices, n={len(ssim_tissue)})')
    axes[1, 1].set_xlabel('SSIM')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].axvline(np.mean(ssim_tissue), color='red', linestyle='--', label=f'Mean={np.mean(ssim_tissue):.4f}')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metric_distributions.png"), dpi=150)
    plt.close()
    print(f"Distribution plots saved.")


def main():
    us_pats = sorted(os.listdir(UNDERSAMPLED_DIR))
    fs_pats = sorted(os.listdir(FULLY_SAMPLED_DIR))
    common = sorted(list(set(us_pats) & set(fs_pats)))
    random.seed(RANDOM_SEED)
    random.shuffle(common)
    n_total = len(common)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    test_patients = common[n_train+n_val:]

    test_dataset = BraTSSliceDataset(test_patients, UNDERSAMPLED_DIR, FULLY_SAMPLED_DIR)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = UNet(1, 1).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    psnr_all, ssim_all = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            p, s = compute_psnr_ssim(outputs, targets)
            psnr_all.append(p)
            ssim_all.append(s)

    psnr_all = np.array(psnr_all)
    ssim_all = np.array(ssim_all)

    tissue_mask = psnr_all <= BACKGROUND_PSNR_THRESH
    psnr_tissue = psnr_all[tissue_mask]
    ssim_tissue = ssim_all[tissue_mask]
 
    with open(os.path.join(OUTPUT_DIR, "metrics.txt"), 'w') as f:
        f.write("=== All Slices ===\n")
        f.write(f"Count: {len(psnr_all)}\n")
        f.write(f"PSNR Mean: {np.mean(psnr_all):.4f} dB\n")
        f.write(f"SSIM Mean: {np.mean(ssim_all):.4f}\n\n")
        f.write("=== Tissue Slices (PSNR <= 72 dB) ===\n")
        f.write(f"Count: {len(psnr_tissue)}\n")
        f.write(f"PSNR Mean: {np.mean(psnr_tissue):.4f} dB\n")
        f.write(f"SSIM Mean: {np.mean(ssim_tissue):.4f}\n")
        f.write(f"Max PSNR in test set: {np.max(psnr_all):.4f} dB\n")

    csv_path = os.path.join(OUTPUT_DIR, "psnr_ssim_raw.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Index", "PSNR_dB", "SSIM"])
        for idx, (p, s) in enumerate(zip(psnr_all, ssim_all)):
            writer.writerow([idx, f"{p:.6f}", f"{s:.6f}"])

    tissue_indices = np.where(tissue_mask)[0]
    if len(tissue_indices) < 5:
        sample_idx = random.sample(range(len(test_dataset)), min(5, len(test_dataset)))
    else:
        sample_idx = random.sample(list(tissue_indices), 5)

    fig, axes = plt.subplots(5, 3, figsize=(9, 15))
    collected = {}
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if i in sample_idx and i not in collected:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                inp = inputs.cpu().squeeze().numpy()
                out = outputs.cpu().squeeze().numpy()
                tar = targets.cpu().squeeze().numpy()
                row = sample_idx.index(i)
                axes[row, 0].imshow(inp, cmap='gray')
                axes[row, 0].set_title("Aliased Input")
                axes[row, 0].axis('off')
                axes[row, 1].imshow(out, cmap='gray')
                axes[row, 1].set_title("Reconstruction")
                axes[row, 1].axis('off')
                axes[row, 2].imshow(tar, cmap='gray')
                axes[row, 2].set_title("Ground Truth")
                axes[row, 2].axis('off')
                collected[i] = True
            if len(collected) == 5:
                break
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "reconstruction_samples.png"))
    plt.close()

    save_metric_distributions(psnr_all, ssim_all, tissue_mask, OUTPUT_DIR)

    print(f"All deliverables saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()