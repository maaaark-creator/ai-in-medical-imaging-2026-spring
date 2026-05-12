from __future__ import annotations

import torch

from model import GatedDuDoRNet, fft2c


def main() -> None:
    torch.manual_seed(42)
    batch, height, width = 2, 128, 128
    undersampled_t2 = torch.rand(batch, 1, height, width)
    t1 = torch.rand(batch, 1, height, width)
    mask = torch.zeros(batch, 1, height, width)
    mask[:, :, height // 2 - 8 : height // 2 + 8, :] = 1.0
    measured_kspace = fft2c(undersampled_t2) * mask

    model = GatedDuDoRNet(
        num_cascades=2,
        features=(16, 32, 64),
        use_kspace_refinement=True,
    )
    output = model(undersampled_t2, t1, mask, measured_kspace)
    print(f"output shape: {tuple(output.shape)}")
    print(f"output range: [{output.min().item():.4f}, {output.max().item():.4f}]")


if __name__ == "__main__":
    main()
