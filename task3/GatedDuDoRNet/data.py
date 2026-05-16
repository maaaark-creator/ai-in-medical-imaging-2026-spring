from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
DEFAULT_ARCHIVE_ROOT = REPO_ROOT / "archive"
DEFAULT_SPLIT_JSON = MODULE_DIR / "splits_seed42.json"


@dataclass(frozen=True)
class SplitConfig:
    archive_root: str = str(DEFAULT_ARCHIVE_ROOT)
    output_json: str = str(DEFAULT_SPLIT_JSON)
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    seed: int = 42


def find_complete_cases(archive_root: Path) -> list[str]:
    case_ids: list[str] = []
    for case_dir in sorted(path for path in archive_root.iterdir() if path.is_dir()):
        case_id = case_dir.name
        t1_path = case_dir / f"{case_id}-t1n.nii"
        t2_path = case_dir / f"{case_id}-t2w.nii"
        if t1_path.exists() and t2_path.exists():
            case_ids.append(case_id)
    return case_ids


def split_cases(
    case_ids: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, list[str]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0.")

    shuffled = list(case_ids)
    random.Random(seed).shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def save_split_json(config: SplitConfig) -> dict[str, list[str]]:
    archive_root = Path(config.archive_root)
    case_ids = find_complete_cases(archive_root)
    splits = split_cases(
        case_ids,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )
    payload = {
        "config": asdict(config),
        "num_cases": {name: len(ids) for name, ids in splits.items()},
        "splits": splits,
    }
    output_path = Path(config.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return splits


def load_split_json(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["splits"]


def normalize_slice(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    nonzero = image[image > 0]
    if nonzero.size == 0:
        return np.zeros_like(image, dtype=np.float32)

    low, high = np.percentile(nonzero, [1.0, 99.5])
    image = np.clip(image, low, high)
    if high > low:
        image = (image - low) / (high - low)
    return image.astype(np.float32)


def fft2c_np(image: np.ndarray) -> np.ndarray:
    shifted = np.fft.ifftshift(image)
    kspace = np.fft.fft2(shifted, norm="ortho")
    return np.fft.fftshift(kspace)


def ifft2c_np(kspace: np.ndarray) -> np.ndarray:
    shifted = np.fft.ifftshift(kspace)
    image = np.fft.ifft2(shifted, norm="ortho")
    return np.fft.fftshift(image)


def generate_variable_density_mask(
    shape: tuple[int, int],
    acceleration: float = 5.0,
    center_fraction: float = 0.10,
    sigma: float = 0.28,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    if acceleration <= 1.0:
        raise ValueError("acceleration must be greater than 1.0.")

    height, width = shape
    total = height * width
    target_samples = int(round(total / acceleration))
    mask = np.zeros((height, width), dtype=bool)

    center_h = max(4, int(round(height * center_fraction)))
    center_w = max(4, int(round(width * center_fraction)))
    row_start = (height - center_h) // 2
    col_start = (width - center_w) // 2
    mask[row_start : row_start + center_h, col_start : col_start + center_w] = True

    remaining = target_samples - int(mask.sum())
    if remaining < 0:
        raise ValueError("center_fraction is too large for the requested acceleration.")

    yy, xx = np.mgrid[0:height, 0:width]
    yy = (yy - (height - 1) / 2.0) / (height / 2.0)
    xx = (xx - (width - 1) / 2.0) / (width / 2.0)
    radius = np.sqrt(xx**2 + yy**2)
    density = np.exp(-(radius**2) / (2.0 * sigma**2)) + 0.01
    density[mask] = 0.0

    if remaining > 0:
        flat_density = density.reshape(-1)
        chosen = rng.choice(
            flat_density.size,
            size=remaining,
            replace=False,
            p=flat_density / flat_density.sum(),
        )
        mask.reshape(-1)[chosen] = True
    return mask.astype(np.float32)


class BraTSMultiModalKSpaceDataset(Dataset):
    """Patient-level split dataset for T1-guided T2w reconstruction.

    The dataset returns one 2D slice at a time and generates the Task 3
    k-space quantities on the fly:
        fully sampled T2w -> FFT -> mask -> measured k-space -> IFFT image.
    """

    def __init__(
        self,
        archive_root: str | Path,
        case_ids: list[str],
        acceleration: float = 5.0,
        center_fraction: float = 0.10,
        sigma: float = 0.28,
        min_nonzero_fraction: float = 0.005,
        seed: int = 42,
    ) -> None:
        if torch is None:
            raise ModuleNotFoundError(
                "PyTorch is required to build BraTSMultiModalKSpaceDataset. "
                "Install torch before running train.py or using the Dataset."
            )
        try:
            import nibabel as nib
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "nibabel is required to load BraTS NIfTI files. "
                "Install nibabel before running train.py or using the Dataset."
            ) from exc

        self.archive_root = Path(archive_root)
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.sigma = sigma
        self.min_nonzero_fraction = min_nonzero_fraction
        self.seed = seed
        self.cases: list[tuple[str, Path, Path, int]] = []
        self.slice_index: list[tuple[int, int]] = []

        for case_id in case_ids:
            case_dir = self.archive_root / case_id
            t1_path = case_dir / f"{case_id}-t1n.nii"
            t2_path = case_dir / f"{case_id}-t2w.nii"
            if not t1_path.exists() or not t2_path.exists():
                continue

            t1_img = nib.load(str(t1_path))
            t2_img = nib.load(str(t2_path))
            num_slices = min(t1_img.shape[2], t2_img.shape[2])
            case_idx = len(self.cases)
            self.cases.append((case_id, t1_path, t2_path, num_slices))

            for slice_z in range(num_slices):
                if self.min_nonzero_fraction > 0.0:
                    t2_slice = np.asarray(t2_img.dataobj[:, :, slice_z], dtype=np.float32)
                    nonzero_fraction = float(np.mean(t2_slice > 0))
                    if nonzero_fraction < self.min_nonzero_fraction:
                        continue
                self.slice_index.append((case_idx, slice_z))

    def __len__(self) -> int:
        return len(self.slice_index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        vol_idx, slice_z = self.slice_index[idx]
        case_id, t1_path, t2_path, _ = self.cases[vol_idx]

        try:
            import nibabel as nib
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "nibabel is required to load BraTS NIfTI files."
            ) from exc

        t1_img = nib.load(str(t1_path))
        t2_img = nib.load(str(t2_path))
        t1_slice = np.asarray(t1_img.dataobj[:, :, slice_z], dtype=np.float32)
        t2_slice = np.asarray(t2_img.dataobj[:, :, slice_z], dtype=np.float32)

        t1 = normalize_slice(t1_slice)
        target_t2 = normalize_slice(t2_slice)

        rng = np.random.default_rng(self.seed + idx)
        mask = generate_variable_density_mask(
            target_t2.shape,
            acceleration=self.acceleration,
            center_fraction=self.center_fraction,
            sigma=self.sigma,
            rng=rng,
        )

        full_t2_kspace = fft2c_np(target_t2)
        measured_kspace = full_t2_kspace * mask
        undersampled_t2 = np.abs(ifft2c_np(measured_kspace)).astype(np.float32)
        t1_kspace = fft2c_np(t1)

        measured_kspace_ri = np.stack(
            [measured_kspace.real, measured_kspace.imag], axis=0
        ).astype(np.float32)
        t1_kspace_ri = np.stack([t1_kspace.real, t1_kspace.imag], axis=0).astype(np.float32)

        return {
            "case_id": case_id,
            "slice_z": slice_z,
            "undersampled_t2": torch.from_numpy(undersampled_t2).unsqueeze(0),
            "t1": torch.from_numpy(t1).unsqueeze(0),
            "target_t2": torch.from_numpy(target_t2).unsqueeze(0),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "measured_kspace": torch.from_numpy(measured_kspace_ri),
            "t1_kspace": torch.from_numpy(t1_kspace_ri),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create patient-level 7:1:2 splits.")
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE_ROOT))
    parser.add_argument("--output-json", default=str(DEFAULT_SPLIT_JSON))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SplitConfig(
        archive_root=args.archive_root,
        output_json=args.output_json,
        seed=args.seed,
    )
    splits = save_split_json(config)
    print("Saved split file:", config.output_json)
    for name, ids in splits.items():
        print(f"{name}: {len(ids)} cases")


if __name__ == "__main__":
    main()
