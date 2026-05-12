from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_complex(kspace: torch.Tensor) -> torch.Tensor:
    """Accept native complex tensors or two-channel real tensors."""
    if torch.is_complex(kspace):
        return kspace
    if kspace.size(1) != 2:
        raise ValueError("Real-valued k-space must have shape [B, 2, H, W].")
    return torch.complex(kspace[:, 0], kspace[:, 1]).unsqueeze(1)


def _match_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.size(1) != 1:
        mask = mask[:, :1]
    return mask.to(device=reference.device, dtype=reference.real.dtype)


def fft2c(image: torch.Tensor) -> torch.Tensor:
    """Centered orthonormal 2D FFT for image tensors shaped [B, C, H, W]."""
    image = torch.fft.ifftshift(image, dim=(-2, -1))
    kspace = torch.fft.fft2(image, dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift(kspace, dim=(-2, -1))


def ifft2c(kspace: torch.Tensor) -> torch.Tensor:
    """Centered orthonormal 2D inverse FFT."""
    kspace = torch.fft.ifftshift(kspace, dim=(-2, -1))
    image = torch.fft.ifft2(kspace, dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift(image, dim=(-2, -1))


class DataConsistencyLayer(nn.Module):
    """Replace sampled k-space locations with the measured values."""

    def __init__(self, blend: float = 1.0) -> None:
        super().__init__()
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend must be in [0, 1].")
        self.blend = blend

    def forward(
        self,
        image: torch.Tensor,
        measured_kspace: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        measured_kspace = _to_complex(measured_kspace)
        mask = _match_mask(mask, measured_kspace)

        pred_kspace = fft2c(image)
        hard_dc = mask * measured_kspace + (1.0 - mask) * pred_kspace
        if self.blend < 1.0:
            hard_dc = self.blend * hard_dc + (1.0 - self.blend) * pred_kspace
        return ifft2c(hard_dc).abs().clamp(0.0, 1.0)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedFusionBlock(nn.Module):
    """Replace DuDoRNet-style concatenation with gated T1-prior fusion."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.mix = DoubleConv(channels * 2, channels)

    def forward(self, t2_feat: torch.Tensor, t1_feat: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([t2_feat, t1_feat], dim=1))
        gated_t1 = gate * t1_feat
        return self.mix(torch.cat([t2_feat, gated_t1], dim=1))


class GatedFusionUNet(nn.Module):
    """Image-domain restoration network with gated T1 image prior."""

    def __init__(
        self,
        in_ch: int = 1,
        guide_ch: int = 1,
        out_ch: int = 1,
        features: Sequence[int] = (32, 64, 128, 256),
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.pool = nn.MaxPool2d(2, 2)

        self.t2_stem = DoubleConv(in_ch, features[0])
        self.t1_stem = DoubleConv(guide_ch, features[0])
        self.stem_fusion = GatedFusionBlock(features[0])

        self.downs = nn.ModuleList()
        in_channels = features[0]
        for feat in features[1:]:
            self.downs.append(DoubleConv(in_channels, feat))
            in_channels = feat

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        self.ups = nn.ModuleList()
        for feat in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feat * 2, feat, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feat * 2, feat))

        self.final_conv = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, t2: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        t2_feat = self.t2_stem(t2)
        t1_feat = self.t1_stem(t1)
        x = self.stem_fusion(t2_feat, t1_feat)

        skips = [x]
        for down in self.downs:
            x = self.pool(x)
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
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)

        update = self.final_conv(x)
        if self.residual:
            return torch.clamp(t2 + update, min=0.0, max=1.0)
        return torch.clamp(update, min=0.0, max=1.0)


def _complex_to_channels(kspace: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(kspace):
        raise ValueError("Expected native complex k-space.")
    return torch.cat([kspace.real, kspace.imag], dim=1)


def _channels_to_complex(channels: torch.Tensor) -> torch.Tensor:
    if channels.size(1) != 2:
        raise ValueError("Expected two channels: real and imaginary.")
    return torch.complex(channels[:, :1], channels[:, 1:2])


class GatedKSpaceRefinementNet(nn.Module):
    """K-space restoration network with gated T1 k-space prior."""

    def __init__(self, channels: int = 32, depth: int = 3) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(4, 2, kernel_size=1),
            nn.Sigmoid(),
        )
        layers: list[nn.Module] = [
            nn.Conv2d(4, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(max(depth - 2, 0)):
            layers.extend(
                [
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(channels, 2, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, t2_kspace: torch.Tensor, t1_kspace: torch.Tensor) -> torch.Tensor:
        t2_channels = _complex_to_channels(t2_kspace)
        t1_channels = _complex_to_channels(t1_kspace)
        gate = self.gate(torch.cat([t2_channels, t1_channels], dim=1))
        gated_t1 = gate * t1_channels
        fused = torch.cat([t2_channels, gated_t1], dim=1)
        residual = self.net(fused)
        refined = t2_channels + residual
        return _channels_to_complex(refined)


class GatedDuDoRNetCascade(nn.Module):
    """One DuDoRNet-style block with gated T1 fusion in both domains."""

    def __init__(
        self,
        features: Sequence[int],
        dc_blend: float,
        use_kspace_refinement: bool,
    ) -> None:
        super().__init__()
        self.image_net = GatedFusionUNet(features=features)
        self.dc = DataConsistencyLayer(blend=dc_blend)
        self.use_kspace_refinement = use_kspace_refinement
        self.kspace_net = GatedKSpaceRefinementNet() if use_kspace_refinement else None

    def forward(
        self,
        image: torch.Tensor,
        t1: torch.Tensor,
        t1_kspace: torch.Tensor,
        measured_kspace: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Image-domain restoration: use gated fusion instead of concat([x, T1]).
        image = self.image_net(image, t1)
        image = self.dc(image, measured_kspace, mask)

        if self.kspace_net is None:
            return image

        # K-space restoration: use gated fusion instead of concat([k, kT1]).
        refined_kspace = self.kspace_net(fft2c(image), t1_kspace)
        measured_kspace = _to_complex(measured_kspace)
        mask = _match_mask(mask, measured_kspace)
        refined_kspace = mask * measured_kspace + (1.0 - mask) * refined_kspace
        return ifft2c(refined_kspace).abs().clamp(0.0, 1.0)


class GatedDuDoRNet(nn.Module):
    """DuDoRNet-style unrolled reconstruction with gated T1-prior fusion.

    The architecture follows the basic DuDoRNet recurrent block:
        image-domain restoration -> DC -> k-space-domain restoration -> DC.

    The difference is the T1 prior fusion strategy. Wherever DuDoRNet feeds
    T1 information by channel-wise concatenation, this implementation uses a
    gated fusion module instead:
        image domain: gate(x, xT1) * xT1
        k-space domain: gate(k, kT1) * kT1

    Args:
        num_cascades: Number of denoising + data-consistency iterations.
        features: U-Net channel widths. Smaller values train faster.
        dc_blend: 1.0 gives hard data consistency at measured k-space samples.
        use_kspace_refinement: Enables a gated T1-prior k-space residual branch.
        share_cascade_weights: If true, recurrent blocks share the same cascade
            parameters, matching the parameter-sharing spirit of DuDoRNet.

    Forward inputs:
        undersampled_t2: magnitude aliased T2 image, shape [B, 1, H, W].
        t1: fully sampled T1n guide image, shape [B, 1, H, W].
        mask: k-space sampling mask, shape [B, 1, H, W] or [B, H, W].
        measured_kspace: undersampled complex T2 k-space. It can be native complex
            [B, 1, H, W] or real two-channel [B, 2, H, W]. If omitted, the model
            falls back to fft2c(undersampled_t2), which is useful only for smoke tests.
        t1_kspace: optional T1 k-space prior. If omitted, fft2c(t1) is used.
    """

    def __init__(
        self,
        num_cascades: int = 4,
        features: Sequence[int] = (32, 64, 128, 256),
        dc_blend: float = 1.0,
        use_kspace_refinement: bool = True,
        share_cascade_weights: bool = True,
    ) -> None:
        super().__init__()
        if num_cascades < 1:
            raise ValueError("num_cascades must be >= 1.")
        self.num_cascades = num_cascades
        self.share_cascade_weights = share_cascade_weights
        first_cascade = GatedDuDoRNetCascade(
            features=features,
            dc_blend=dc_blend,
            use_kspace_refinement=use_kspace_refinement,
        )
        if share_cascade_weights:
            self.cascade = first_cascade
            self.cascades = None
        else:
            self.cascade = None
            self.cascades = nn.ModuleList(
                [
                    first_cascade,
                    *[
                        GatedDuDoRNetCascade(
                            features=features,
                            dc_blend=dc_blend,
                            use_kspace_refinement=use_kspace_refinement,
                        )
                        for _ in range(num_cascades - 1)
                    ],
                ]
            )

    def forward(
        self,
        undersampled_t2: torch.Tensor,
        t1: torch.Tensor,
        mask: torch.Tensor,
        measured_kspace: torch.Tensor | None = None,
        t1_kspace: torch.Tensor | None = None,
        return_intermediate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if measured_kspace is None:
            measured_kspace = fft2c(undersampled_t2)
        if t1_kspace is None:
            t1_kspace = fft2c(t1)
        else:
            t1_kspace = _to_complex(t1_kspace)

        x = undersampled_t2
        intermediates: list[torch.Tensor] = []
        cascade_iterable = (
            [self.cascade] * self.num_cascades
            if self.share_cascade_weights
            else self.cascades
        )
        for cascade in cascade_iterable:
            x = cascade(x, t1, t1_kspace, measured_kspace, mask)
            if return_intermediate:
                intermediates.append(x)

        if return_intermediate:
            return x, intermediates
        return x
