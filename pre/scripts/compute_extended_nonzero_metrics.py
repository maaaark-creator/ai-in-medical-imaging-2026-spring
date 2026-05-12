from __future__ import annotations

import csv
import gzip
import json
import math
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PRE_DIR = SCRIPT_DIR.parent
GIT_DIR = PRE_DIR.parent
ASSET_DIR = PRE_DIR / "assets"
TABLE_DIR = ASSET_DIR / "tables"

LINE_REPO = GIT_DIR / "ai-in-medical-imaging-2026-spring -linemask"
LINE_TASK2 = LINE_REPO / "outputs" / "task2"
ARCHIVE_ROOT = GIT_DIR / "archive"
LINE_UNDERSAMPLED_ROOT = GIT_DIR / "undersampled_raw_data_t2w_vertical_line_r5"

THRESHOLD = 0.001
BATCH_SIZE = 8

MODEL_DIRS = {
    "2D U-Net": LINE_TASK2 / "baseline_2d_unet_vertical_line_r5_all_p99_train",
    "2.5D U-Net bf32": LINE_TASK2 / "final_25d_unet_vertical_line_r5_ctx3_all_p99_full",
    "2.5D U-Net bf64": LINE_TASK2 / "final_25d_unet_vertical_line_r5_ctx3_all_p99_bf64_full",
    "2.5D Residual ResNet": LINE_TASK2 / "exp_25d_resnet_vertical_line_r5_ctx3_all_shared_p99_train",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_nii_array(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        header = f.read(348)
        sizeof_hdr_le = struct.unpack("<i", header[:4])[0]
        sizeof_hdr_be = struct.unpack(">i", header[:4])[0]
        endian = "<" if sizeof_hdr_le == 348 else ">"
        dims = struct.unpack(endian + "8h", header[40:56])
        ndim = int(dims[0])
        shape = tuple(int(v) for v in dims[1 : ndim + 1])
        datatype = struct.unpack(endian + "h", header[70:72])[0]
        vox_offset = int(round(struct.unpack(endian + "f", header[108:112])[0]))
        scl_slope = struct.unpack(endian + "f", header[112:116])[0]
        scl_inter = struct.unpack(endian + "f", header[116:120])[0]
        dtype_map = {
            2: np.uint8,
            4: np.int16,
            8: np.int32,
            16: np.float32,
            64: np.float64,
            256: np.int8,
            512: np.uint16,
            768: np.uint32,
            1024: np.int64,
            1280: np.uint64,
        }
        dtype = np.dtype(dtype_map[datatype]).newbyteorder(endian)
        f.seek(vox_offset)
        count = int(np.prod(shape))
        data = np.frombuffer(f.read(count * dtype.itemsize), dtype=dtype, count=count)
    arr = data.reshape(shape, order="F").astype(np.float32, copy=False)
    if scl_slope not in (0.0, 1.0) or scl_inter != 0.0:
        slope = 1.0 if scl_slope == 0.0 else float(scl_slope)
        arr = arr * slope + float(scl_inter)
    return arr


def find_t2w(root: Path, patient_id: str) -> Path:
    for suffix in (".nii", ".nii.gz"):
        path = root / patient_id / f"{patient_id}-t2w{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(patient_id)


def robust_scale(volume: np.ndarray) -> float:
    nonzero = volume[volume > 0]
    if nonzero.size == 0:
        return 1.0
    scale = float(np.percentile(nonzero, 99.0))
    return scale if scale > 0 else 1.0


def normalize_shared(slice_2d: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.asarray(slice_2d, dtype=np.float32) / max(scale, 1e-8), 0.0, 1.0)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class DoubleConv2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet2D(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, features: list[int] | None = None) -> None:
        super().__init__()
        features = features or [64, 128, 256, 512]
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        for feat in features:
            self.downs.append(DoubleConv2D(in_ch, feat))
            in_ch = feat
        self.bottleneck = DoubleConv2D(features[-1], features[-1] * 2)
        for feat in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feat * 2, feat, kernel_size=2, stride=2))
            self.ups.append(DoubleConv2D(feat * 2, feat))
        self.final_conv = nn.Conv2d(features[0], out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)
        return self.final_conv(x)


class DoubleConv25D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet25D(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: tuple[int, ...] = (32, 64, 128, 256)) -> None:
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        current_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv25D(current_channels, feature))
            current_channels = feature
        self.bottleneck = DoubleConv25D(features[-1], features[-1] * 2)
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv25D(feature * 2, feature))
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx // 2]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)
        return self.final_conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ResNet25D(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 64, num_blocks: int = 12, dilations: tuple[int, ...] = (1, 2, 4, 1)) -> None:
        super().__init__()
        self.center_index = in_channels // 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(base_channels), base_channels),
            nn.SiLU(inplace=True),
        )
        self.body = nn.Sequential(
            *[ResidualBlock(base_channels, dilation=dilations[idx % len(dilations)]) for idx in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(base_channels), base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        center = x[:, self.center_index : self.center_index + 1]
        residual = self.head(self.body(self.stem(x)))
        return center + residual


def load_models(device: torch.device) -> dict[str, tuple[nn.Module, str, int]]:
    models: dict[str, tuple[nn.Module, str, int]] = {}

    model2d = UNet2D().to(device)
    state2d = torch.load(MODEL_DIRS["2D U-Net"] / "best_unet.pth", map_location=device, weights_only=False)
    model2d.load_state_dict(state2d)
    model2d.eval()
    models["2D U-Net"] = (model2d, "2d", 1)

    for label in ("2.5D U-Net bf32", "2.5D U-Net bf64"):
        ckpt = torch.load(MODEL_DIRS[label] / "best_unet25d.pth", map_location=device, weights_only=False)
        features = tuple(ckpt.get("features", (32, 64, 128, 256)))
        context_slices = int(ckpt.get("config", {}).get("context_slices", 3))
        model = UNet25D(in_channels=context_slices, features=features).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        models[label] = (model, "25d", context_slices)

    ckpt = torch.load(MODEL_DIRS["2.5D Residual ResNet"] / "best_resnet25d.pth", map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    context_slices = int(config.get("context_slices", 3))
    model = ResNet25D(
        in_channels=context_slices,
        base_channels=int(ckpt.get("base_channels", config.get("base_channels", 64))),
        num_blocks=int(ckpt.get("num_blocks", config.get("num_blocks", 12))),
        dilations=tuple(ckpt.get("dilations", config.get("dilations_tuple", (1, 2, 4, 1)))),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    models["2.5D Residual ResNet"] = (model, "25d", context_slices)
    return models


def make_inputs(us_vol: np.ndarray, indices: list[int], scale: float, mode: str, context_slices: int) -> np.ndarray:
    if mode == "2d":
        return np.stack([normalize_shared(us_vol[:, :, z], scale)[None] for z in indices], axis=0)
    half = context_slices // 2
    samples = []
    for z in indices:
        chans = []
        for offset in range(-half, half + 1):
            zz = min(max(z + offset, 0), us_vol.shape[2] - 1)
            chans.append(normalize_shared(us_vol[:, :, zz], scale))
        samples.append(np.stack(chans, axis=0))
    return np.stack(samples, axis=0)


def load_existing_metric_vectors() -> dict[str, pd.DataFrame]:
    vectors: dict[str, pd.DataFrame] = {}
    resnet_df = pd.read_csv(MODEL_DIRS["2.5D Residual ResNet"] / "per_slice_metrics.csv")
    nonzero_mask = resnet_df["is_nonblank"].astype(bool)
    vectors["Input / before recon"] = pd.DataFrame(
        {
            "psnr": resnet_df.loc[nonzero_mask, "before_psnr"].to_numpy(float),
            "ssim": resnet_df.loc[nonzero_mask, "before_ssim"].to_numpy(float),
        }
    )
    for label in ("2.5D U-Net bf32", "2.5D U-Net bf64", "2.5D Residual ResNet"):
        df = pd.read_csv(MODEL_DIRS[label] / "per_slice_metrics.csv")
        tissue = df[df["is_nonblank"].astype(bool)]
        vectors[label] = pd.DataFrame(
            {
                "psnr": tissue["after_psnr"].to_numpy(float),
                "ssim": tissue["after_ssim"].to_numpy(float),
            }
        )

    df2 = pd.read_csv(LINE_TASK2 / "baseline_2d_unet_vertical_line_r5_all_p99_eval_target_nonzero" / "psnr_ssim_raw.csv")
    tissue2 = df2[df2["target_nonzero_fraction"].astype(float) >= THRESHOLD]
    vectors["2D U-Net"] = pd.DataFrame(
        {
            "psnr": tissue2["PSNR_dB"].to_numpy(float),
            "ssim": tissue2["SSIM"].to_numpy(float),
        }
    )
    return vectors


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "variance": float(np.var(values, ddof=0)),
    }


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    split = read_json(MODEL_DIRS["2.5D Residual ResNet"] / "split_patients.json")
    test_patients = list(split["test"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(device)

    extended_rows: list[dict[str, object]] = []
    before_rows: list[dict[str, object]] = []

    with torch.no_grad():
        for patient_idx, patient_id in enumerate(test_patients, start=1):
            us_vol = read_nii_array(find_t2w(LINE_UNDERSAMPLED_ROOT, patient_id))
            fs_vol = read_nii_array(find_t2w(ARCHIVE_ROOT, patient_id))
            n_slices = min(us_vol.shape[2], fs_vol.shape[2])
            us_vol = us_vol[:, :, :n_slices]
            fs_vol = fs_vol[:, :, :n_slices]
            scale = robust_scale(fs_vol)

            nonzero_indices = [
                z for z in range(n_slices)
                if float(np.count_nonzero(fs_vol[:, :, z]) / fs_vol[:, :, z].size) >= THRESHOLD
            ]
            targets = [normalize_shared(fs_vol[:, :, z], scale) for z in nonzero_indices]

            for z, target in zip(nonzero_indices, targets):
                center = normalize_shared(us_vol[:, :, z], scale)
                diff = center - target
                before_rows.append(
                    {
                        "model": "Input / before recon",
                        "patient_id": patient_id,
                        "slice_index": z,
                        "mae": float(np.mean(np.abs(diff))),
                        "rmse": float(np.sqrt(np.mean(diff * diff))),
                    }
                )

            for label, (model, mode, context_slices) in models.items():
                for start in range(0, len(nonzero_indices), BATCH_SIZE):
                    batch_indices = nonzero_indices[start : start + BATCH_SIZE]
                    batch_targets = np.stack(targets[start : start + BATCH_SIZE], axis=0)
                    x = make_inputs(us_vol, batch_indices, scale, mode, context_slices)
                    tensor = torch.from_numpy(x.astype(np.float32)).to(device)
                    pred = model(tensor).detach().cpu().numpy()[:, 0]
                    pred = np.clip(pred, 0.0, 1.0)
                    diff = pred - batch_targets
                    mae = np.mean(np.abs(diff), axis=(1, 2))
                    rmse = np.sqrt(np.mean(diff * diff, axis=(1, 2)))
                    for z, one_mae, one_rmse in zip(batch_indices, mae, rmse):
                        extended_rows.append(
                            {
                                "model": label,
                                "patient_id": patient_id,
                                "slice_index": z,
                                "mae": float(one_mae),
                                "rmse": float(one_rmse),
                            }
                        )
            if patient_idx % 10 == 0:
                print(f"Processed {patient_idx}/{len(test_patients)} patients")

    errors = pd.DataFrame(before_rows + extended_rows)
    errors_path = TABLE_DIR / "task2_nonzero_mae_rmse_per_slice.csv"
    errors.to_csv(errors_path, index=False)

    metric_vectors = load_existing_metric_vectors()
    rows = []
    for model, df in metric_vectors.items():
        err = errors[errors["model"] == model]
        psnr_stats = summarize(df["psnr"].to_numpy(float))
        ssim_stats = summarize(df["ssim"].to_numpy(float))
        mae_stats = summarize(err["mae"].to_numpy(float))
        rmse_stats = summarize(err["rmse"].to_numpy(float))
        rows.append(
            {
                "model": model,
                "n_nonzero_slices": int(len(df)),
                "psnr_mean": psnr_stats["mean"],
                "psnr_median": psnr_stats["median"],
                "psnr_variance": psnr_stats["variance"],
                "ssim_mean": ssim_stats["mean"],
                "ssim_median": ssim_stats["median"],
                "ssim_variance": ssim_stats["variance"],
                "mae_mean": mae_stats["mean"],
                "mae_median": mae_stats["median"],
                "mae_variance": mae_stats["variance"],
                "rmse_mean": rmse_stats["mean"],
                "rmse_median": rmse_stats["median"],
                "rmse_variance": rmse_stats["variance"],
            }
        )
    summary = pd.DataFrame(rows)
    order = ["Input / before recon", "2D U-Net", "2.5D U-Net bf32", "2.5D U-Net bf64", "2.5D Residual ResNet"]
    summary["model"] = pd.Categorical(summary["model"], categories=order, ordered=True)
    summary = summary.sort_values("model").astype({"model": str})
    summary.to_csv(TABLE_DIR / "task2_nonzero_extended_metrics_summary.csv", index=False)

    md = [
        "| Model | n | PSNR mean | PSNR median | PSNR var | SSIM mean | SSIM median | SSIM var | MAE mean | RMSE mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        md.append(
            f"| {row['model']} | {int(row['n_nonzero_slices'])} | "
            f"{row['psnr_mean']:.2f} | {row['psnr_median']:.2f} | {row['psnr_variance']:.2f} | "
            f"{row['ssim_mean']:.3f} | {row['ssim_median']:.3f} | {row['ssim_variance']:.6f} | "
            f"{row['mae_mean']:.5f} | {row['rmse_mean']:.5f} |"
        )
    (TABLE_DIR / "task2_nonzero_extended_metrics_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {errors_path}")
    print(f"Wrote {TABLE_DIR / 'task2_nonzero_extended_metrics_summary.csv'}")


if __name__ == "__main__":
    main()
