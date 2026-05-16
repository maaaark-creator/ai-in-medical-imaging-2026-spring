from __future__ import annotations

import torch
import torch.nn as nn


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ResNet25D(nn.Module):
    """2.5D residual image-to-image network for T2w alias removal."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        num_blocks: int = 12,
        dilations: tuple[int, ...] = (1, 2, 4, 1),
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1.")
        if not dilations:
            raise ValueError("dilations must not be empty.")

        self.center_index = in_channels // 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(base_channels), base_channels),
            nn.SiLU(inplace=True),
        )
        self.body = nn.Sequential(
            *[
                ResidualBlock(base_channels, dilation=dilations[idx % len(dilations)])
                for idx in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(base_channels), base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        center = x[:, self.center_index : self.center_index + 1]
        residual = self.head(self.body(self.stem(x)))
        return center + residual
