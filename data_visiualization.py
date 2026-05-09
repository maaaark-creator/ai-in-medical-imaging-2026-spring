from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
from skimage.measure import marching_cubes


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_DATA_ROOT = PROJECT_ROOT.parent / "archive"
LEGACY_DATA_ROOT = Path("raw_data")
DEFAULT_CASE_ID = "BraTS-GLI-00000-000"
LOCAL_OUTPUT = PROJECT_ROOT / "outputs" / "visualization" / "t1_tumor_3d_visualization.html"
LEGACY_OUTPUT = Path("outputs") / "t1_tumor_3d_visualization.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a rough 3D brain structure from fused T1/T2 with tumor overlay.")
    parser.add_argument(
        "--path-profile",
        choices=["local", "legacy"],
        default="local",
        help="local uses ../archive next to the repository; legacy keeps the original raw_data default.",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Root directory containing BraTS case folders.")
    parser.add_argument("--case-id", type=str, default=DEFAULT_CASE_ID, help="Case folder to visualize.")
    parser.add_argument("--t1", type=Path, default=None, help="Path to the T1 volume (.nii or .nii.gz).")
    parser.add_argument("--t2", type=Path, default=None, help="Path to the T2 volume (.nii or .nii.gz).")
    parser.add_argument("--seg", type=Path, default=None, help="Path to the tumor segmentation volume.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="HTML path for the interactive 3D visualization.",
    )
    parser.add_argument(
        "--brain-percentile",
        type=float,
        default=82.0,
        help="Percentile used to extract a rough fused-brain surface.",
    )
    parser.add_argument(
        "--t1-weight",
        type=float,
        default=0.5,
        help="Weight for T1 in the fused background volume.",
    )
    parser.add_argument(
        "--t2-weight",
        type=float,
        default=0.5,
        help="Weight for T2 in the fused background volume.",
    )
    parser.add_argument(
        "--tumor-labels",
        type=int,
        nargs="*",
        default=[1, 2, 3],
        help="Segmentation labels to merge into the tumor mask.",
    )
    parser.add_argument("--no-open", action="store_true", help="Save the HTML but do not open it automatically.")
    args = parser.parse_args()
    resolve_paths(args)
    return args


def resolve_paths(args: argparse.Namespace) -> None:
    data_root = args.data_root
    output = args.output
    if args.path_profile == "legacy":
        data_root = LEGACY_DATA_ROOT if data_root is None else data_root
        output = LEGACY_OUTPUT if output is None else output
    else:
        data_root = LOCAL_DATA_ROOT if data_root is None else data_root
        output = LOCAL_OUTPUT if output is None else output

    case_dir = data_root / args.case_id
    args.t1 = args.t1 if args.t1 is not None else case_dir / f"{args.case_id}-t1n.nii"
    args.t2 = args.t2 if args.t2 is not None else case_dir / f"{args.case_id}-t2w.nii"
    args.seg = args.seg if args.seg is not None else case_dir / f"{args.case_id}-seg.nii"
    args.output = output


def load_volume(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(), dtype=np.float32)
    spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
    return data, spacing


def normalize_nonzero(data: np.ndarray) -> np.ndarray:
    result = np.zeros_like(data, dtype=np.float32)
    nonzero = data > 0
    if not np.any(nonzero):
        return result

    values = data[nonzero]
    low = float(np.percentile(values, 1))
    high = float(np.percentile(values, 99))
    if high <= low:
        result[nonzero] = 1.0
        return result

    clipped = np.clip(data, low, high)
    result[nonzero] = ((clipped[nonzero] - low) / (high - low)).astype(np.float32)
    return result


def fuse_modalities(t1_data: np.ndarray, t2_data: np.ndarray, t1_weight: float, t2_weight: float) -> np.ndarray:
    if t1_data.shape != t2_data.shape:
        raise ValueError(f"T1 shape {t1_data.shape} does not match T2 shape {t2_data.shape}.")

    total_weight = t1_weight + t2_weight
    if total_weight <= 0:
        raise ValueError("t1_weight + t2_weight must be greater than 0.")

    t1_norm = normalize_nonzero(t1_data)
    t2_norm = normalize_nonzero(t2_data)
    return (t1_weight * t1_norm + t2_weight * t2_norm) / total_weight


def build_mesh(mask: np.ndarray, spacing: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    verts, faces, _, _ = marching_cubes(mask.astype(np.float32), level=0.5, spacing=spacing)
    return verts, faces


def add_mesh(fig: go.Figure, verts: np.ndarray, faces: np.ndarray, name: str, color: str, opacity: float) -> None:
    fig.add_trace(
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name=name,
            color=color,
            opacity=opacity,
            flatshading=True,
        )
    )


def build_brain_mask(background_data: np.ndarray, percentile: float) -> tuple[np.ndarray, float]:
    nonzero = background_data[background_data > 0]
    if nonzero.size == 0:
        raise ValueError("The fused background volume is empty.")

    threshold = float(np.percentile(nonzero, percentile))
    mask = background_data >= threshold
    return mask, threshold


def build_tumor_mask(seg_data: np.ndarray, labels: list[int]) -> np.ndarray:
    mask = np.isin(seg_data, labels)
    if np.count_nonzero(mask) == 0:
        raise ValueError(f"No tumor voxels found for labels {labels}.")
    return mask


def create_figure(
    t1_data: np.ndarray,
    t2_data: np.ndarray,
    seg_data: np.ndarray,
    spacing: tuple[float, float, float],
    brain_percentile: float,
    t1_weight: float,
    t2_weight: float,
    tumor_labels: list[int],
) -> tuple[go.Figure, float]:
    if t1_data.shape != seg_data.shape:
        raise ValueError(f"T1 shape {t1_data.shape} does not match segmentation shape {seg_data.shape}.")
    if t2_data.shape != seg_data.shape:
        raise ValueError(f"T2 shape {t2_data.shape} does not match segmentation shape {seg_data.shape}.")

    background_data = fuse_modalities(t1_data, t2_data, t1_weight, t2_weight)
    brain_mask, used_threshold = build_brain_mask(background_data, brain_percentile)
    tumor_mask = build_tumor_mask(seg_data, tumor_labels)

    brain_verts, brain_faces = build_mesh(brain_mask, spacing)
    tumor_verts, tumor_faces = build_mesh(tumor_mask, spacing)

    fig = go.Figure()
    add_mesh(fig, brain_verts, brain_faces, "Fused T1/T2 brain structure", "#a8dadc", 0.18)
    add_mesh(fig, tumor_verts, tumor_faces, "Tumor", "#d62828", 0.78)

    fig.update_layout(
        title="Fused T1/T2 Brain Structure with Tumor Overlay",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig, used_threshold


def main() -> None:
    args = parse_args()
    t1_data, t1_spacing = load_volume(args.t1)
    t2_data, t2_spacing = load_volume(args.t2)
    seg_data, seg_spacing = load_volume(args.seg)

    if t1_spacing != seg_spacing:
        raise ValueError(f"T1 spacing {t1_spacing} does not match segmentation spacing {seg_spacing}.")
    if t2_spacing != seg_spacing:
        raise ValueError(f"T2 spacing {t2_spacing} does not match segmentation spacing {seg_spacing}.")

    figure, used_threshold = create_figure(
        t1_data=t1_data,
        t2_data=t2_data,
        seg_data=seg_data,
        spacing=t1_spacing,
        brain_percentile=args.brain_percentile,
        t1_weight=args.t1_weight,
        t2_weight=args.t2_weight,
        tumor_labels=args.tumor_labels,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot(figure, filename=str(args.output), auto_open=not args.no_open)

    print(f"T1 file: {args.t1}")
    print(f"T2 file: {args.t2}")
    print(f"Seg file: {args.seg}")
    print(f"Volume shape: {t1_data.shape}")
    print(f"Voxel spacing: {t1_spacing}")
    print(f"T1/T2 weights: {args.t1_weight:.2f}/{args.t2_weight:.2f}")
    print(f"Brain threshold percentile: {args.brain_percentile}")
    print(f"Brain threshold value: {used_threshold:.3f}")
    print(f"Saved interactive 3D view to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
