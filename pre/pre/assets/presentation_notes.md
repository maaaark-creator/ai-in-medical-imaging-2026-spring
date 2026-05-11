# Task 2 presentation asset notes

Main interpretation:

- Use vertical-line R=5 results as final Task 2 evidence.
- Treat point-wise mask experiments as preliminary pipeline validation only.
- Do not claim that 2.5D context is universally better than 2D.
- Strongest supported claim: residual artifact correction performs best under Cartesian line undersampling.

Main tissue-slice results:

- Input / before recon: PSNR 28.54, SSIM 0.721
- 2D U-Net: PSNR 36.36, SSIM 0.973, gain +7.82 dB / +0.252
- 2.5D U-Net: PSNR 35.38, SSIM 0.978, gain +6.84 dB / +0.257
- 2.5D Residual ResNet: PSNR 38.70, SSIM 0.981, gain +10.16 dB / +0.260

Useful caveats:

- 2D U-Net and 2.5D U-Net are not fully capacity-matched.
- Current Task 2 models are image-domain models without explicit k-space data consistency.
- Tissue-slice metrics are primary because background slices can bias all-slice scores.
