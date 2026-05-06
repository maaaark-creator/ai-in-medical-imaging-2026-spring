import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from collections import defaultdict
from tqdm import tqdm

UNDERSAMPLED_DIR = os.path.expanduser(
    "~/Desktop/project1_without_rawdata/undersampled_raw_data_t2w_r5"
)
FULLY_SAMPLED_DIR = os.path.expanduser("~/Downloads/dataset/archive")
OUTPUT_DIR = os.path.expanduser("~/Desktop/project1_without_rawdata/task2_final_deliverables)
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
EARLY_STOP_PATIENCE = 2
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
NUM_WORKERS = 4
BACKGROUND_PSNR_THRESH = 56.0

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
            if not os.path.exists(us_path):
                print(f"Missing undersampled: {us_path}, skipping {pid}")
                continue
            if not os.path.exists(fs_path):
                print(f"Missing fully sampled: {fs_path}, skipping {pid}")
                continue
            try:
                us_vol = nib.load(us_path).get_fdata().astype(np.float32)
                fs_vol = nib.load(fs_path).get_fdata().astype(np.float32)
            except Exception as e:
                print(f"Error loading {pid}: {e}")
                continue
            num_slices = min(us_vol.shape[2], fs_vol.shape[2])
            vol_idx = len(self.volumes)
            self.volumes.append((us_vol, fs_vol))
            for s in range(num_slices):
                self.slice_index.append((vol_idx, s))
        print(
            f"Loaded {len(self.volumes)} patients, total slices: {len(self.slice_index)}"
        )
    def __len__(self):
        return len(self.slice_index)
    def __getitem__(self, idx):
        vol_idx, slice_z = self.slice_index[idx]
        us_vol, fs_vol = self.volumes[vol_idx]
        img_input = us_vol[:, :, slice_z]
        img_target = fs_vol[:, :, slice_z]

        def norm_slice(s):
            mn, mx = s.min(), s.max()
            if mx > mn:
                return (s - mn) / (mx - mn)
            return np.zeros_like(s)

        img_input = norm_slice(img_input)
        img_target = norm_slice(img_target)

        img_input = torch.from_numpy(img_input).unsqueeze(0)
        img_target = torch.from_numpy(img_target).unsqueeze(0)
        return img_input, img_target

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
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
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        for feat in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feat * 2, feat, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feat * 2, feat))

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
            skip = skip_connections[idx // 2]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)
        return self.final_conv(x)

def compute_psnr_ssim(pred, target):
    pred = pred.squeeze().cpu().numpy()
    target = target.squeeze().cpu().numpy()
    psnr = peak_signal_noise_ratio(target, pred, data_range=1.0)
    ssim = structural_similarity(target, pred, data_range=1.0)
    return float(psnr), float(ssim)

def main():
    undersampled_patients = sorted(os.listdir(UNDERSAMPLED_DIR))
    fully_sampled_patients = sorted(os.listdir(FULLY_SAMPLED_DIR))
    common_patients = sorted(
        list(set(undersampled_patients) & set(fully_sampled_patients))
    )
    print(f"Total patients with both data: {len(common_patients)}")
    random.seed(42)
    random.shuffle(common_patients)
    n_total = len(common_patients)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    train_patients = common_patients[:n_train]
    val_patients = common_patients[n_train : n_train + n_val]
    test_patients = common_patients[n_train + n_val :]
    print(f"Train patients: {len(train_patients)}")
    print(f"Val   patients: {len(val_patients)}")
    print(f"Test  patients: {len(test_patients)}")
    train_dataset = BraTSSliceDataset(
        train_patients, UNDERSAMPLED_DIR, FULLY_SAMPLED_DIR
    )
    val_dataset = BraTSSliceDataset(val_patients, UNDERSAMPLED_DIR, FULLY_SAMPLED_DIR)
    test_dataset = BraTSSliceDataset(test_patients, UNDERSAMPLED_DIR, FULLY_SAMPLED_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Train slices: {len(train_dataset)}")
    print(f"Val   slices: {len(val_dataset)}")
    print(f"Test  slices: {len(test_dataset)}")
    model = UNet(in_ch=1, out_ch=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=False
    )
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):

        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({"loss": loss.item()})
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        print(
            f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_unet.pth"))
            print("  -> Saved best model")
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement for {epochs_without_improvement} epoch(s)")
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    plt.close()
    print("\n=== Testing Best Model ===")
    model.load_state_dict(
        torch.load(os.path.join(OUTPUT_DIR, "best_unet.pth"), map_location=device)
    )
    model.eval()
    psnr_all, ssim_all = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            psnr, ssim = compute_psnr_ssim(outputs, targets)
            psnr_all.append(psnr)
            ssim_all.append(ssim)
    psnr_all = np.array(psnr_all)
    ssim_all = np.array(ssim_all)
    avg_psnr_full = np.mean(psnr_all)
    avg_ssim_full = np.mean(ssim_all)
    tissue_mask = psnr_all <= BACKGROUND_PSNR_THRESH
    psnr_tissue = psnr_all[tissue_mask]
    ssim_tissue = ssim_all[tissue_mask]
    if len(psnr_tissue) == 0:
        print("Warning: All slices were classified as background! Check threshold.")
        avg_psnr_tissue = 0.0
        avg_ssim_tissue = 0.0
    else:
        avg_psnr_tissue = np.mean(psnr_tissue)
        avg_ssim_tissue = np.mean(ssim_tissue)
    print(
        f"\nAll slices ({len(psnr_all)}): PSNR = {avg_psnr_full:.4f} dB, SSIM = {avg_ssim_full:.4f}"
    )
    print(
        f"Tissue slices only ({len(psnr_tissue)}): PSNR = {avg_psnr_tissue:.4f} dB, SSIM = {avg_ssim_tissue:.4f}"
    )

    with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
        f.write("=== All Slices ===\n")
        f.write(f"Count  : {len(psnr_all)}\n")
        f.write(f"PSNR dB (mean): {avg_psnr_full:.4f}\n")
        f.write(f"SSIM (mean)   : {avg_ssim_full:.4f}\n\n")
        f.write("=== Tissue Slices (PSNR <= 56.0 dB) ===\n")
        f.write(f"Count  : {len(psnr_tissue)}\n")
        f.write(f"PSNR dB (mean): {avg_psnr_tissue:.4f}\n")
        f.write(f"SSIM (mean)   : {avg_ssim_tissue:.4f}\n\n")
        f.write(
            "Per-slice PSNR (all): ["
            + ", ".join([f"{v:.6f}" for v in psnr_all])
            + "]\n"
        )
        f.write(
            "Per-slice SSIM (all): ["
            + ", ".join([f"{v:.6f}" for v in ssim_all])
            + "]\n"
        )
    tissue_indices = np.where(tissue_mask)[0]
    if len(tissue_indices) == 0:
        sample_indices = random.sample(
            range(len(test_dataset)), min(5, len(test_dataset))
        )
    else:
        sample_indices = random.sample(
            list(tissue_indices), min(5, len(tissue_indices))
        )
    fig, axes = plt.subplots(
        len(sample_indices), 3, figsize=(9, 3 * len(sample_indices))
    )
    if len(sample_indices) == 1:
        axes = [axes]
    collected = {idx: None for idx in sample_indices}
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if i in collected:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                inp_np = inputs.cpu().squeeze().numpy()
                out_np = outputs.cpu().squeeze().numpy()
                tar_np = targets.cpu().squeeze().numpy()
                psnr_val, ssim_val = compute_psnr_ssim(outputs, targets)
                row = sample_indices.index(i)
                axes[row][0].imshow(inp_np, cmap="gray")
                axes[row][0].set_title("Aliased Input")
                axes[row][0].axis("off")
                axes[row][1].imshow(out_np, cmap="gray")
                axes[row][1].set_title(f"Recon (PSNR:{psnr_val:.1f})")
                axes[row][1].axis("off")
                axes[row][2].imshow(tar_np, cmap="gray")
                axes[row][2].set_title("Ground Truth")
                axes[row][2].axis("off")
            if len(collected) == len(sample_indices):
                break
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "reconstruction_samples.png"))
    plt.close()
    print(f"\nAll results saved to {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()