# Task 2 presentation asset notes

Main interpretation:

- Use vertical-line R=5 results as final Task 2 evidence.
- Treat point-wise mask experiments as a separate artifact-regime comparison.
- Do not claim that 2.5D context is universally better than 2D.
- The wider 2.5D U-Net bf64 is a capacity ablation: it improves over bf32, but only modestly.
- Strongest supported claim: residual artifact correction performs best under Cartesian line undersampling.

Main tissue-slice results:

- Input / before recon: PSNR 28.54, SSIM 0.721
- 2D U-Net: PSNR 36.36, SSIM 0.973, gain +7.82 dB / +0.252
- 2.5D U-Net: PSNR 35.38, SSIM 0.978, gain +6.84 dB / +0.257
- 2.5D U-Net bf64: PSNR 35.79, SSIM 0.979, gain +7.25 dB / +0.258
- 2.5D Residual ResNet: PSNR 38.70, SSIM 0.981, gain +10.16 dB / +0.260

Useful caveats:

- 2D U-Net and 2.5D U-Net are not fully capacity-matched.
- The bf64 ablation reduces the capacity concern but does not overturn the residual-correction conclusion.
- Current Task 2 models are image-domain models without explicit k-space data consistency.
- Tissue-slice metrics are primary because background slices can bias all-slice scores.
