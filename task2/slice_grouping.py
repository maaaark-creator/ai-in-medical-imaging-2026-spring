from __future__ import annotations

from pathlib import Path

import numpy as np

SLICE_GROUPING_CHOICES = ("target_nonzero", "psnr_threshold")
DEFAULT_SLICE_GROUPING = "target_nonzero"
DEFAULT_TARGET_NONZERO_THRESHOLD = 0.001
DEFAULT_BACKGROUND_PSNR_THRESHOLD = 56.0


def default_eval_output_dir(output_root: Path, slice_grouping: str) -> Path:
    return Path(output_root) / f"baseline_2d_unet_vertical_line_r5_nonzero_p99_eval_{slice_grouping}"


def target_nonzero_fraction(slice_2d: np.ndarray) -> float:
    array = np.asarray(slice_2d)
    return float(np.count_nonzero(array) / array.size) if array.size else 0.0


def build_nonblank_mask(
    psnr_values: np.ndarray,
    target_nonzero_fractions: np.ndarray,
    slice_grouping: str,
    target_nonzero_threshold: float,
    background_psnr_threshold: float,
) -> np.ndarray:
    if slice_grouping == "target_nonzero":
        return np.asarray(target_nonzero_fractions) >= target_nonzero_threshold
    if slice_grouping == "psnr_threshold":
        return np.asarray(psnr_values) <= background_psnr_threshold
    raise ValueError(f"Unsupported slice grouping: {slice_grouping}")


def grouping_title(
    slice_grouping: str,
    target_nonzero_threshold: float,
    background_psnr_threshold: float,
) -> str:
    if slice_grouping == "target_nonzero":
        return (
            "Non-blank slices "
            f"(target nonzero fraction >= {target_nonzero_threshold:g})"
        )
    if slice_grouping == "psnr_threshold":
        return f"Tissue slices (PSNR <= {background_psnr_threshold:g} dB)"
    raise ValueError(f"Unsupported slice grouping: {slice_grouping}")
