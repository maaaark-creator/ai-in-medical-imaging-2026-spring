from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


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


def validate_context_slices(context_slices: int) -> None:
    if context_slices < 1 or context_slices % 2 == 0:
        raise ValueError("--context-slices must be a positive odd integer, e.g. 3 or 5.")


def validate_normalization(normalization: str) -> None:
    if normalization not in {"independent", "shared"}:
        raise ValueError("--normalization must be either 'independent' or 'shared'.")


def find_t2w_path(root: Path, patient_id: str) -> Path:
    candidates = [
        root / patient_id / f"{patient_id}-t2w.nii",
        root / patient_id / f"{patient_id}-t2w.nii.gz",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def normalize_independent(slice_2d: np.ndarray) -> np.ndarray:
    slice_2d = np.asarray(slice_2d, dtype=np.float32)
    mn = float(slice_2d.min())
    mx = float(slice_2d.max())
    if mx > mn:
        return (slice_2d - mn) / (mx - mn)
    return np.zeros_like(slice_2d, dtype=np.float32)


def normalize_shared(slice_2d: np.ndarray, scale: float) -> np.ndarray:
    slice_2d = np.asarray(slice_2d, dtype=np.float32)
    if scale <= 0.0:
        return np.zeros_like(slice_2d, dtype=np.float32)
    return np.clip(slice_2d / scale, 0.0, 1.0).astype(np.float32)


def nonzero_fraction(slice_2d: np.ndarray) -> float:
    return float(np.count_nonzero(slice_2d) / slice_2d.size)


def robust_shared_scale(volume: np.ndarray, percentile: float) -> float:
    nonzero = np.asarray(volume[volume > 0], dtype=np.float32)
    if nonzero.size == 0:
        return 1.0
    scale = float(np.percentile(nonzero, percentile))
    return scale if scale > 0.0 else 1.0


class BraTS25DResNetSliceDataset(Dataset):
    def __init__(
        self,
        patient_ids: list[str],
        undersampled_root: Path,
        fully_sampled_root: Path,
        context_slices: int = 3,
        slice_filter: str = "nonzero",
        normalization: str = "shared",
        robust_percentile: float = 99.0,
        blank_threshold: float = 0.001,
        cache_size: int = 2,
        return_metadata: bool = False,
        desc: str = "Building dataset",
    ) -> None:
        validate_context_slices(context_slices)
        validate_normalization(normalization)
        if slice_filter not in {"all", "nonzero"}:
            raise ValueError("--slice-filter must be either 'all' or 'nonzero'.")
        if not (0.0 < robust_percentile <= 100.0):
            raise ValueError("--robust-percentile must be in (0, 100].")

        self.patient_ids = patient_ids
        self.undersampled_root = Path(undersampled_root)
        self.fully_sampled_root = Path(fully_sampled_root)
        self.context_slices = context_slices
        self.slice_filter = slice_filter
        self.normalization = normalization
        self.robust_percentile = robust_percentile
        self.blank_threshold = blank_threshold
        self.cache_size = max(cache_size, 0)
        self.return_metadata = return_metadata
        self.records: list[PatientRecord] = []
        self.slice_index: list[tuple[int, int, float]] = []
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.missing_patients: list[str] = []
        self.shape_mismatches: list[dict[str, Any]] = []

        self._build_index(desc)

    def _build_index(self, desc: str) -> None:
        for patient_id in tqdm(self.patient_ids, desc=desc):
            us_path = find_t2w_path(self.undersampled_root, patient_id)
            fs_path = find_t2w_path(self.fully_sampled_root, patient_id)
            if not us_path.exists() or not fs_path.exists():
                self.missing_patients.append(patient_id)
                continue

            try:
                us_img = nib.load(str(us_path))
                fs_img = nib.load(str(fs_path))
            except Exception:
                self.missing_patients.append(patient_id)
                continue

            if us_img.shape[:2] != fs_img.shape[:2]:
                self.shape_mismatches.append(
                    {
                        "patient_id": patient_id,
                        "undersampled_shape": tuple(us_img.shape),
                        "fully_sampled_shape": tuple(fs_img.shape),
                    }
                )
                continue

            num_slices = min(us_img.shape[2], fs_img.shape[2])
            fs_data = np.asanyarray(fs_img.dataobj)
            shared_scale = robust_shared_scale(fs_data[:, :, :num_slices], self.robust_percentile)
            patient_idx = len(self.records)
            kept = 0
            filtered = 0

            for slice_z in range(num_slices):
                frac = nonzero_fraction(fs_data[:, :, slice_z])
                keep = self.slice_filter == "all" or frac >= self.blank_threshold
                if keep:
                    self.slice_index.append((patient_idx, slice_z, frac))
                    kept += 1
                else:
                    filtered += 1

            self.records.append(
                PatientRecord(
                    patient_id=patient_id,
                    undersampled_path=us_path,
                    fully_sampled_path=fs_path,
                    num_slices=num_slices,
                    total_slices=num_slices,
                    kept_slices=kept,
                    filtered_slices=filtered,
                    shared_scale=shared_scale,
                )
            )

    def __len__(self) -> int:
        return len(self.slice_index)

    def _load_pair(self, patient_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if patient_idx in self._cache:
            self._cache.move_to_end(patient_idx)
            return self._cache[patient_idx]

        record = self.records[patient_idx]
        us_vol = nib.load(str(record.undersampled_path)).get_fdata(dtype=np.float32)
        fs_vol = nib.load(str(record.fully_sampled_path)).get_fdata(dtype=np.float32)

        if self.cache_size > 0:
            self._cache[patient_idx] = (us_vol, fs_vol)
            self._cache.move_to_end(patient_idx)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return us_vol, fs_vol

    def _normalize(self, slice_2d: np.ndarray, shared_scale: float) -> np.ndarray:
        if self.normalization == "independent":
            return normalize_independent(slice_2d)
        return normalize_shared(slice_2d, shared_scale)

    def __getitem__(self, idx: int):
        patient_idx, slice_z, frac = self.slice_index[idx]
        record = self.records[patient_idx]
        us_vol, fs_vol = self._load_pair(patient_idx)

        half = self.context_slices // 2
        input_slices = []
        for offset in range(-half, half + 1):
            z = min(max(slice_z + offset, 0), record.num_slices - 1)
            input_slices.append(self._normalize(us_vol[:, :, z], record.shared_scale))

        target = self._normalize(fs_vol[:, :, slice_z], record.shared_scale)
        inputs = torch.from_numpy(np.stack(input_slices, axis=0).astype(np.float32))
        target_tensor = torch.from_numpy(target[None, :, :].astype(np.float32))

        if not self.return_metadata:
            return inputs, target_tensor

        center_input = input_slices[half][None, :, :].astype(np.float32)
        metadata = {
            "patient_id": record.patient_id,
            "slice_index": int(slice_z),
            "target_nonzero_fraction": float(frac),
            "center_input": torch.from_numpy(center_input),
            "shared_scale": float(record.shared_scale),
        }
        return inputs, target_tensor, metadata

    def stats(self) -> dict[str, Any]:
        total = int(sum(record.total_slices for record in self.records))
        kept = int(sum(record.kept_slices for record in self.records))
        filtered = int(sum(record.filtered_slices for record in self.records))
        per_patient = []
        for record in self.records:
            item = record.__dict__.copy()
            item["undersampled_path"] = str(item["undersampled_path"])
            item["fully_sampled_path"] = str(item["fully_sampled_path"])
            per_patient.append(item)
        return {
            "patients_requested": len(self.patient_ids),
            "patients_loaded": len(self.records),
            "missing_or_unreadable_patients": self.missing_patients,
            "shape_mismatches": self.shape_mismatches,
            "slice_filter": self.slice_filter,
            "normalization": self.normalization,
            "robust_percentile": self.robust_percentile,
            "blank_threshold": self.blank_threshold,
            "total_slices": total,
            "kept_slices": kept,
            "filtered_slices": filtered,
            "filtered_fraction": filtered / total if total else 0.0,
            "per_patient": per_patient,
        }
