from __future__ import annotations

import argparse
import csv
import random
import sys
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_: object):
        return iterable

# ================= 路径配置 =================
BASE_DIR = Path(__file__).resolve().parent

# 数据绝对路径
DEFAULT_UNDERSAMPLED_DIR = Path("/data/new_disk7/zrd/ai_medical/undersampled_raw_data_t2w_vertical_line_r5")
DEFAULT_FULLY_SAMPLED_DIR = Path("/data/new_disk7/zrd/ai_medical/archive")

# 读取 DenseNet 的训练输出
TRAIN_OUTPUT_DIR = BASE_DIR / "task2_densenet_deliverables_new"
DEFAULT_CHECKPOINT = TRAIN_OUTPUT_DIR / "best_pure_densenet_new.pt"

# 测试结果存放目录
DEFAULT_TEST_OUT_DIR = BASE_DIR / "task2_densenet_test_results"

# ================= 1. 数据集逻辑 (带黑图过滤) =================
def find_t2w_path(root: Path, patient_id: str) -> Path:
    candidates = [root / patient_id / f"{patient_id}-t2w.nii", root / patient_id / f"{patient_id}-t2w.nii.gz"]
    for p in candidates:
        if p.exists(): return p
    return candidates[0]

def robust_shared_scale(volume: np.ndarray, percentile: float=99.0) -> float:
    nonzero = np.asarray(volume[volume > 0], dtype=np.float32)
    if nonzero.size == 0: return 1.0
    scale = float(np.percentile(nonzero, percentile))
    return scale if scale > 0.0 else 1.0

class BraTS2DTestDataset(Dataset):
    def __init__(self, patient_ids, undersampled_root, fully_sampled_root, blank_threshold=0.001):
        self.us_root = Path(undersampled_root)
        self.fs_root = Path(fully_sampled_root)
        self.blank_threshold = blank_threshold
        
        self.records = []
        self.slice_index = []
        
        filtered_count = 0
        kept_count = 0

        print(f"\nScanning {len(patient_ids)} test patients and filtering blank slices...")
        for patient_id in tqdm(patient_ids, desc="Loading Test Set"):
            us_path = find_t2w_path(self.us_root, patient_id)
            fs_path = find_t2w_path(self.fs_root, patient_id)
            if not us_path.exists() or not fs_path.exists(): continue
            
            us_img = nib.load(str(us_path))
            fs_img = nib.load(str(fs_path))
            if us_img.shape[:2] != fs_img.shape[:2]: continue

            num_slices = min(us_img.shape[2], fs_img.shape[2])
            fs_data = np.asanyarray(fs_img.dataobj)
            shared_scale = robust_shared_scale(fs_data)
            
            p_idx = len(self.records)
            for z in range(num_slices):
                slice_data = fs_data[:, :, z]
                frac = float(np.count_nonzero(slice_data) / slice_data.size) if slice_data.size else 0.0
                
                if frac >= self.blank_threshold:
                    self.slice_index.append((p_idx, z, frac))
                    kept_count += 1
                else:
                    filtered_count += 1
            
            self.records.append({
                "patient_id": patient_id, "us_path": us_path, "fs_path": fs_path, "scale": shared_scale
            })
            
        print(f" -> Kept {kept_count} informative slices. Filtered out {filtered_count} blank/dark slices.")

    def __len__(self): return len(self.slice_index)

    def __getitem__(self, idx):
        p_idx, z, frac = self.slice_index[idx]
        record = self.records[p_idx]
        
        us_vol = nib.load(str(record["us_path"])).get_fdata(dtype=np.float32)
        fs_vol = nib.load(str(record["fs_path"])).get_fdata(dtype=np.float32)
        
        scale = record["scale"]
        inp = np.clip(us_vol[:, :, z] / scale, 0.0, 1.0)
        tgt = np.clip(fs_vol[:, :, z] / scale, 0.0, 1.0)
        
        return {
            "input": torch.from_numpy(inp[None, :, :]).float(),
            "target": torch.from_numpy(tgt[None, :, :]).float(),
            "patient_id": record["patient_id"],
            "slice_z": z
        }

# ================= 2. Pure DenseNet 模型定义 =================
def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs): return conv(relu(norm(torch.cat(inputs, 1))))
    return bn_function

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate
    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        new_features = self.conv2(self.relu2(self.norm2(bn_function(*prev_features))))
        if self.drop_rate > 0: new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)
    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children(): features.append(layer(*features))
        return torch.cat(features, 1)

class PureDenseNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_init_features=32, growth_rate=16, block_config=(4, 4, 4, 4), drop_rate=0):
        super(PureDenseNet, self).__init__()
        self.init_conv = nn.Sequential(nn.Conv2d(in_channels, num_init_features, 3, 1, 1, bias=False), nn.BatchNorm2d(num_init_features), nn.ReLU(inplace=True))
        self.features = nn.Sequential()
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size=4, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = nn.Sequential(nn.BatchNorm2d(num_features), nn.ReLU(inplace=True), nn.Conv2d(num_features, num_features // 2, 1, 1, bias=False))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.final_conv = nn.Conv2d(num_features, out_channels, 3, 1, 1)

    def forward(self, x):
        return self.final_conv(self.features(self.init_conv(x)))

# ================= 3. 指标计算与评估 =================
def compute_batch_metrics(pred: torch.Tensor, target: torch.Tensor):
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(pred - target), dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-12))
    
    c1, c2 = 0.01**2, 0.03**2
    kernel = torch.ones((1, 1, 7, 7), device=pred.device, dtype=pred.dtype) / 49.0
    mu_x, mu_y = F.conv2d(pred, kernel, padding=3), F.conv2d(target, kernel, padding=3)
    mu_x2, mu_y2, mu_xy = mu_x.pow(2), mu_y.pow(2), mu_x * mu_y
    sigma_x = F.conv2d(pred * pred, kernel, padding=3) - mu_x2
    sigma_y = F.conv2d(target * target, kernel, padding=3) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel, padding=3) - mu_xy
    ssim = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2))
    ssim = ssim.mean(dim=(1, 2, 3))
    
    return mse.sum().item(), rmse.sum().item(), mae.sum().item(), psnr.sum().item(), ssim.sum().item()

def evaluate(model, loader, device):
    model.eval()
    t_mse, t_rmse, t_mae, t_psnr, t_ssim, count = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    samples = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                preds = model(inputs)
            
            preds_fp32 = preds.float()
            targets_fp32 = targets.float()
            
            mse, rmse, mae, psnr, ssim = compute_batch_metrics(preds_fp32, targets_fp32)
            bs = targets.size(0)
            
            t_mse += mse; t_rmse += rmse; t_mae += mae; t_psnr += psnr; t_ssim += ssim
            count += bs
            
            if len(samples) < 10:
                for i in range(bs):
                    if len(samples) >= 10: break
                    samples.append({
                        "pid": batch["patient_id"][i], "z": batch["slice_z"][i].item(),
                        "inp": inputs[i, 0].cpu().numpy(), "tgt": targets_fp32[i, 0].cpu().numpy(), "pred": preds_fp32[i, 0].cpu().numpy()
                    })

    return {
        "mse": t_mse/count, "rmse": t_rmse/count, "mae": t_mae/count,
        "psnr": t_psnr/count, "ssim": t_ssim/count
    }, samples

def plot_samples(samples, out_dir):
    try: import matplotlib.pyplot as plt
    except: return
    
    fig, axes = plt.subplots(len(samples), 4, figsize=(15, 3.5 * len(samples)))
    for r, s in enumerate(samples):
        error = np.abs(s["pred"] - s["tgt"])
        axes[r, 0].imshow(s["inp"], cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[r, 0].set_title("1. Undersampled Input")
        axes[r, 1].imshow(s["pred"], cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[r, 1].set_title("2. Pure DenseNet Recon")
        axes[r, 2].imshow(s["tgt"], cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[r, 2].set_title("3. Ground Truth")
        axes[r, 3].imshow(error, cmap="magma", origin="lower", vmin=0, vmax=0.3)
        axes[r, 3].set_title("4. Error Map")
        axes[r, 0].set_ylabel(f"{s['pid']}\nSlice: {s['z']}", fontsize=12, fontweight='bold')
        for c in range(4): axes[r, c].set_xticks([]); axes[r, c].set_yticks([])

    fig.tight_layout()
    fig.savefig(out_dir / "pure_densenet_visualizations.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--blank-threshold", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = DEFAULT_TEST_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    if "split" not in ckpt:
        print("[FATAL] split dictionary not found in checkpoint! Cannot ensure test set isolation.")
        sys.exit(1)
        
    test_patients = ckpt["split"]["test"]
    print(f"Isolated Test Patients: {len(test_patients)}")
    
    dataset = BraTS2DTestDataset(test_patients, DEFAULT_UNDERSAMPLED_DIR, DEFAULT_FULLY_SAMPLED_DIR, args.blank_threshold)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = PureDenseNet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    
    metrics, samples = evaluate(model, loader, device)
    
    print("\n" + "="*50)
    print(" FINAL PURE DENSENET TEST RESULTS (Non-Blank Slices) ")
    print(f"  Total Slices Evaluated : {len(dataset)}")
    print(f"  MSE  : {metrics['mse']:.6f}")
    print(f"  RMSE : {metrics['rmse']:.6f}")
    print(f"  MAE  : {metrics['mae']:.6f}")
    print(f"  PSNR : {metrics['psnr']:.4f} dB")
    print(f"  SSIM : {metrics['ssim']:.4f}")
    print("="*50)
    
    # 写入结果到文件
    with open(out_dir / "test_metrics.txt", "w") as f:
        f.write("Pure DenseNet Test Metrics\n---------------------\n")
        for k, v in metrics.items(): f.write(f"{k.upper()}: {v}\n")
            
    plot_samples(samples, out_dir)
    print(f"\n Visualizations saved to: {out_dir / 'pure_densenet_visualizations.png'}")

if __name__ == "__main__":
    main()