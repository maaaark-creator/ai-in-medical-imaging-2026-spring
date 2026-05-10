from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent

DEFAULT_UNDERSAMPLED_DIR = DATA_ROOT / "undersampled_raw_data_t2w_r5"
DEFAULT_FULLY_SAMPLED_DIR = DATA_ROOT / "archive"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "task2"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

DEFAULT_CONTEXT_SLICES = 3
DEFAULT_SLICE_FILTER = "nonzero"
DEFAULT_BLANK_THRESHOLD = 0.001
DEFAULT_NORM_MODE = "separate"
DEFAULT_ROBUST_PERCENTILE = 99.0
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 30
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_EARLY_STOP_PATIENCE = 5
DEFAULT_SEED = 42
DEFAULT_NUM_WORKERS = 4
DEFAULT_CACHE_SIZE = 2


def expand_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def default_output_dir(context_slices: int, slice_filter: str, norm_mode: str = DEFAULT_NORM_MODE) -> Path:
    suffix = f"25d_unet_context{context_slices}_{slice_filter}"
    if norm_mode != DEFAULT_NORM_MODE:
        suffix = f"{suffix}_{norm_mode}"
    return DEFAULT_OUTPUT_ROOT / suffix


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_common_patients(undersampled_dir: Path, fully_sampled_dir: Path) -> list[str]:
    undersampled = {p.name for p in undersampled_dir.iterdir() if p.is_dir()}
    fully_sampled = {p.name for p in fully_sampled_dir.iterdir() if p.is_dir()}
    return sorted(undersampled & fully_sampled)


def split_patients(
    patient_ids: list[str],
    seed: int,
    limit_patients: int | None = None,
) -> dict[str, list[str]]:
    patients = list(patient_ids)
    rng = random.Random(seed)
    rng.shuffle(patients)
    if limit_patients is not None:
        patients = patients[: max(limit_patients, 0)]

    n_total = len(patients)
    if n_total == 0:
        return {"train": [], "val": [], "test": []}

    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    if n_total >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        if n_train + n_val >= n_total:
            n_val = max(1, n_total - n_train - 1)
    elif n_total == 2:
        n_train, n_val = 1, 0
    else:
        n_train, n_val = 1, 0

    train = patients[:n_train]
    val = patients[n_train : n_train + n_val]
    test = patients[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
