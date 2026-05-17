from __future__ import annotations

import argparse
import csv
import random
import sys
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

# ================= 路径与全局配置 =================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

# 直接写死绝对路径，最稳妥不会报错
DEFAULT_UNDERSAMPLED_DIR = Path("/data/new_disk7/zrd/ai_medical/undersampled_raw_data_t2w_vertical_line_r5")
DEFAULT_FULLY_SAMPLED_DIR = Path("/data/new_disk7/zrd/ai_medical/archive")
DEFAULT_OUTPUT_DIR = BASE_DIR / "task2_denseunet_deliverables_new"

# ================= 1. 数据结构与辅助函数 (与测试脚本对齐) =================

@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    undersampled_path: Path
    fully_sampled_path: Path
    num_slices: int
    total_slices: int
    kept_slices: int
    filtered_slices: int
    shared_scale: float

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def find_t2w_path(root: Path, patient_id: str) -> Path:
    candidates = [
        root / patient_id / f"{patient_id}-t2w.nii",
        root / patient_id / f"{patient_id}-t2w.nii.gz",
    ]
    for path in candidates:
        if path.exists(): return path
    return candidates[0]

def list_common_patients(undersampled_dir: Path, fully_sampled_dir: Path) -> list[str]:
    undersampled = {p.name for p in undersampled_dir.iterdir() if p.is_dir()}
    fully_sampled = {p.name for p in fully_sampled_dir.iterdir() if p.is_dir()}
    return sorted(undersampled & fully_sampled)

def split_patients(patient_ids: list[str], seed: int) -> dict[str, list[str]]:
    patients = list(patient_ids)
    random.Random(seed).shuffle(patients)
    n_total = len(patients)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.1)
    
    return {
        "train": patients[:n_train],
        "val": patients[n_train : n_train + n_val],
        "test": patients[n_train + n_val :],
    }

# ================= 2. 归一化与 Dataset 类 =================

def normalize_shared(slice_2d: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0.0: return np.zeros_like(slice_2d, dtype=np.float32)
    return np.clip(np.asarray(slice_2d, dtype=np.float32) / scale, 0.0, 1.0).astype(np.float32)

def robust_shared_scale(volume: np.ndarray, percentile: float) -> float:
    nonzero = np.asarray(volume[volume > 0], dtype=np.float32)
    if nonzero.size == 0: return 1.0
    scale = float(np.percentile(nonzero, percentile))
    return scale if scale > 0.0 else 1.0

class BraTS2DSliceDataset(Dataset):
    def __init__(self, patient_ids, undersampled_root, fully_sampled_root, 
                 slice_filter="all", normalization="shared", robust_percentile=99.0, 
                 blank_threshold=0.001, cache_size=4, desc="Loading"):
        self.undersampled_root = Path(undersampled_root)
        self.fully_sampled_root = Path(fully_sampled_root)
        self.slice_filter = slice_filter
        self.normalization = normalization
        self.cache_size = cache_size
        self._cache = OrderedDict()
        self.records = []
        self.slice_index = []

        for patient_id in tqdm(patient_ids, desc=desc):
            us_path = find_t2w_path(self.undersampled_root, patient_id)
            fs_path = find_t2w_path(self.fully_sampled_root, patient_id)
            if not us_path.exists() or not fs_path.exists(): continue
            
            us_img = nib.load(str(us_path))
            fs_img = nib.load(str(fs_path))
            if us_img.shape[:2] != fs_img.shape[:2]: continue

            num_slices = min(us_img.shape[2], fs_img.shape[2])
            fs_data = np.asanyarray(fs_img.dataobj)
            shared_scale = robust_shared_scale(fs_data, robust_percentile)
            
            p_idx = len(self.records)
            for z in range(num_slices):
                self.slice_index.append((p_idx, z))
            
            self.records.append(PatientRecord(patient_id, us_path, fs_path, num_slices, num_slices, num_slices, 0, shared_scale))

    def __len__(self): return len(self.slice_index)

    def _load_pair(self, p_idx):
        if p_idx in self._cache:
            self._cache.move_to_end(p_idx)
            return self._cache[p_idx]
        record = self.records[p_idx]
        us_vol = nib.load(str(record.undersampled_path)).get_fdata(dtype=np.float32)
        fs_vol = nib.load(str(record.fully_sampled_path)).get_fdata(dtype=np.float32)
        if self.cache_size > 0:
            self._cache[p_idx] = (us_vol, fs_vol)
            if len(self._cache) > self.cache_size: self._cache.popitem(last=False)
        return us_vol, fs_vol

    def __getitem__(self, idx):
        p_idx, z = self.slice_index[idx]
        record = self.records[p_idx]
        us_vol, fs_vol = self._load_pair(p_idx)
        
        input_slice = normalize_shared(us_vol[:,:,z], record.shared_scale)
        target_slice = normalize_shared(fs_vol[:,:,z], record.shared_scale)
        
        return torch.from_numpy(input_slice[None,:,:]), torch.from_numpy(target_slice[None,:,:])

# ================= 3. DenseUNet 模型定义 =================

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

class DenseUNetReconstructor(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_init_features=32, growth_rate=16, block_config=(4, 4, 4, 4), drop_rate=0):
        super(DenseUNetReconstructor, self).__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features), nn.ReLU(inplace=True)
        )
        self.encoder_dense, self.encoder_trans = nn.ModuleList(), nn.ModuleList()
        num_features = num_init_features
        self.skip_channels = []
        
        for i in range(len(block_config) - 1):
            num_layers = block_config[i]
            self.encoder_dense.append(_DenseBlock(num_layers, num_features, 4, growth_rate, drop_rate))
            num_features += num_layers * growth_rate
            self.skip_channels.append(num_features)
            self.encoder_trans.append(nn.Sequential(
                nn.BatchNorm2d(num_features), nn.ReLU(inplace=True),
                nn.Conv2d(num_features, num_features // 2, 1, 1, bias=False),
                nn.AvgPool2d(2, 2)
            ))
            num_features //= 2
            
        self.bottleneck = _DenseBlock(block_config[-1], num_features, 4, growth_rate, drop_rate)
        num_features += block_config[-1] * growth_rate
        
        self.decoder_trans_up, self.decoder_dense = nn.ModuleList(), nn.ModuleList()
        up_config = block_config[:-1][::-1]
        skips = self.skip_channels[::-1]
        for i in range(len(up_config)):
            self.decoder_trans_up.append(nn.ConvTranspose2d(num_features, num_features // 2, 2, 2))
            num_features = (num_features // 2) + skips[i]
            self.decoder_dense.append(_DenseBlock(up_config[i], num_features, 4, growth_rate, drop_rate))
            num_features += up_config[i] * growth_rate
            
        self.final_conv = nn.Conv2d(num_features, out_channels, 1, 1)

    def forward(self, x):
        features = self.init_conv(x)
        skip_connections = []
        for db, tr in zip(self.encoder_dense, self.encoder_trans):
            features = db(features)
            skip_connections.append(features)
            features = tr(features)
        features = self.bottleneck(features)
        for tu, db, sk in zip(self.decoder_trans_up, self.decoder_dense, skip_connections[::-1]):
            features = tu(features)
            if features.shape != sk.shape:
                features = F.interpolate(features, size=sk.shape[2:], mode='bilinear', align_corners=False)
            features = torch.cat([features, sk], 1)
            features = db(features)
        return self.final_conv(features)

# ================= 4. 训练逻辑与日志功能 =================

def plot_loss_curve(history: list[dict[str, float]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    epochs = [item["epoch"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    val_losses = [item["val_loss"] for item in history]
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_losses, label="Train Loss (MSE)", marker="o", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Validation Loss (MSE)", marker="o", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("DenseUNet Training Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve_new.png", dpi=180)
    plt.close(fig)

def run_epoch(model, loader, criterion, device, scaler=None, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    for inputs, targets in tqdm(loader, desc="Train" if is_train else "Val"):
        inputs, targets = inputs.to(device), targets.to(device)
        if is_train: optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            preds = model(inputs)
            loss = criterion(preds, targets)
        
        if is_train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * targets.size(0)
    return total_loss / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    
    # 【新增参数】断点续训路径和学习率恢复
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (e.g. best_denseunet_new.pt)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (check your CSV for the LR before crash)")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    model = DenseUNetReconstructor().to(device)
    
    start_epoch = 1
    best_loss = float('inf')
    csv_mode = "w"
    history = []

    # ================= 核心断点续训逻辑 =================
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"\n[INFO] Resuming training from: {resume_path.name}")
            checkpoint = torch.load(resume_path, map_location=device)
            
            # 1. 恢复模型权重
            model.load_state_dict(checkpoint["model_state_dict"])
            
            # 2. 恢复数据划分 (最重要的一步，防止数据泄露)
            split = checkpoint["split"]
            print("[INFO] Patient split successfully restored from checkpoint.")
            
            # 3. 恢复训练进度
            start_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint.get("val_loss", float('inf'))
            csv_mode = "a" # 追加模式，不覆盖之前的日志
            print(f"[INFO] Resuming from Epoch {start_epoch} (Best Val Loss so far: {best_loss:.6f})")
        else:
            print(f"[ERROR] Checkpoint not found at {resume_path}")
            sys.exit(1)
    else:
        # 如果不是续训，正常从头划分数据
        all_patients = list_common_patients(DEFAULT_UNDERSAMPLED_DIR, DEFAULT_FULLY_SAMPLED_DIR)
        split = split_patients(all_patients, args.seed)
    # ====================================================

    train_ds = BraTS2DSliceDataset(split["train"], DEFAULT_UNDERSAMPLED_DIR, DEFAULT_FULLY_SAMPLED_DIR, desc="Train Index")
    val_ds = BraTS2DSliceDataset(split["val"], DEFAULT_UNDERSAMPLED_DIR, DEFAULT_FULLY_SAMPLED_DIR, desc="Val Index")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    criterion = nn.MSELoss()
    
    # 使用传入的 lr 初始化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    scaler = torch.cuda.amp.GradScaler()
    
    log_path = out_dir / "training_log_new.csv"
    
    # 尝试恢复之前 CSV 里的 history 数据用于画图
    if csv_mode == "a" and log_path.exists():
        with log_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                history.append({
                    "epoch": float(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "val_loss": float(row["val_loss"])
                })
    
    with log_path.open(csv_mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if csv_mode == "w":
            writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

        for epoch in range(start_epoch, args.epochs + 1):
            train_loss = run_epoch(model, train_loader, criterion, device, scaler, optimizer)
            val_loss = run_epoch(model, val_loader, criterion, device)
            
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")
            
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{current_lr:.2e}"])
            f.flush()
            
            history.append({
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss
            })
            
            if val_loss < best_loss:
                best_loss = val_loss
                save_path = out_dir / "best_denseunet_new.pt"
                # 现在我们将优化器状态也存进去，以备将来还需要断点续训
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "split": split,
                    "epoch": epoch,
                    "val_loss": val_loss
                }, save_path)
                print(f"   --> New Best Validation Loss! Saved to {save_path.name}")

    torch.save(model.state_dict(), out_dir / "last_denseunet_state_dict_new.pt")
    plot_loss_curve(history, out_dir)
    print("\nTraining Complete!")

if __name__ == "__main__":
    main()