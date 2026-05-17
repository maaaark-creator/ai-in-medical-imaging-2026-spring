from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.view(1, 1, window_size, window_size)


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Differentiable single-channel SSIM loss: 1 - SSIM."""
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape.")
    if pred.size(1) != 1:
        raise ValueError("This SSIM implementation expects single-channel images.")

    pad = window_size // 2
    window = _gaussian_window(window_size, sigma, pred.device, pred.dtype)

    mu_x = F.conv2d(pred, window, padding=pad)
    mu_y = F.conv2d(target, window, padding=pad)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x = F.conv2d(pred * pred, window, padding=pad) - mu_x2
    sigma_y = F.conv2d(target * target, window, padding=pad) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=pad) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    )
    return 1.0 - ssim_map.mean()


class HybridReconstructionLoss(nn.Module):
    """Contrast-preserving reconstruction loss for MRI restoration."""

    def __init__(
        self,
        l1_weight: float = 0.90,
        ssim_weight: float = 0.03,
        low_freq_weight: float = 0.04,
        intensity_weight: float = 0.03,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.low_freq_weight = low_freq_weight
        self.intensity_weight = intensity_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.components(pred, target)["total_loss"]

    def components(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        pred = pred.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)
        mask = (target > 1e-4).to(dtype=pred.dtype)
        mask_count = mask.sum().clamp_min(1.0)

        l1 = (torch.abs(pred - target) * mask).sum() / mask_count
        ssim = ssim_loss(pred, target)
        low_freq = F.l1_loss(
            F.avg_pool2d(pred, kernel_size=9, stride=1, padding=4),
            F.avg_pool2d(target, kernel_size=9, stride=1, padding=4),
        )

        pred_mean = (pred * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        target_mean = (target * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        pred_var = (((pred - pred_mean[:, None, None, None]) * mask) ** 2).sum(
            dim=(1, 2, 3)
        ) / mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        target_var = (((target - target_mean[:, None, None, None]) * mask) ** 2).sum(
            dim=(1, 2, 3)
        ) / mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        intensity = F.l1_loss(pred_mean, target_mean) + F.l1_loss(
            torch.sqrt(pred_var.clamp_min(1e-8)),
            torch.sqrt(target_var.clamp_min(1e-8)),
        )

        weighted_l1 = self.l1_weight * l1
        weighted_ssim = self.ssim_weight * ssim
        weighted_low_freq = self.low_freq_weight * low_freq
        weighted_intensity = self.intensity_weight * intensity
        total = weighted_l1 + weighted_ssim + weighted_low_freq + weighted_intensity
        return {
            "total_loss": total,
            "l1_loss": l1,
            "ssim_loss": ssim,
            "low_freq_loss": low_freq,
            "intensity_loss": intensity,
            "weighted_l1": weighted_l1,
            "weighted_ssim": weighted_ssim,
            "weighted_low_freq": weighted_low_freq,
            "weighted_intensity": weighted_intensity,
        }
