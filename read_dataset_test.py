import argparse
import gzip
import os
import struct
from pathlib import Path
from typing import Dict, Tuple


EXPECTED_SUFFIXES = ["t1c", "t1n", "t2f", "t2w", "seg"]
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_DATA_ROOT = PROJECT_ROOT.parent / "archive"
LEGACY_DATA_ROOT = Path(r"F:\zhujiao\archive")


def read_nifti_shape(file_path: str) -> Tuple[int, ...]:
    open_func = gzip.open if file_path.lower().endswith(".gz") else open
    with open_func(file_path, "rb") as file_obj:
        header = file_obj.read(348)

    if len(header) < 48:
        raise ValueError("File is too small to be a valid NIfTI file.")

    sizeof_hdr = struct.unpack("<I", header[0:4])[0]
    if sizeof_hdr != 348:
        sizeof_hdr = struct.unpack(">I", header[0:4])[0]
        if sizeof_hdr != 348:
            raise ValueError("Invalid NIfTI header size.")
        endian = ">"
    else:
        endian = "<"

    dims = struct.unpack(endian + "8h", header[40:56])
    ndim = dims[0]
    if ndim <= 0:
        raise ValueError("Invalid NIfTI dimension count.")

    shape = tuple(int(value) for value in dims[1 : ndim + 1])
    if not shape:
        raise ValueError("Empty NIfTI shape.")

    return shape


def find_patient_files(patient_dir: str) -> Dict[str, str]:
    matched_files: Dict[str, str] = {}
    for file_name in os.listdir(patient_dir):
        lower_name = file_name.lower()
        if not (lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")):
            continue

        for suffix in EXPECTED_SUFFIXES:
            marker = "-" + suffix + ".nii"
            marker_gz = marker + ".gz"
            if lower_name.endswith(marker) or lower_name.endswith(marker_gz):
                matched_files[suffix] = os.path.join(patient_dir, file_name)
                break

    return matched_files


def get_patient_dirs(root_dir: str):
    patient_dirs = []
    for patient_name in sorted(os.listdir(root_dir)):
        patient_dir = os.path.join(root_dir, patient_name)
        if os.path.isdir(patient_dir):
            patient_dirs.append((patient_name, patient_dir))
    return patient_dirs


def print_results(root_dir: str) -> None:
    patient_dirs = get_patient_dirs(root_dir)
    print("Patient count:", len(patient_dirs))

    if not patient_dirs:
        return

    first_patient_name, first_patient_dir = patient_dirs[0]
    print("First patient:", first_patient_name)

    files = find_patient_files(first_patient_dir)
    for suffix in EXPECTED_SUFFIXES:
        file_path = files.get(suffix)
        if not file_path:
            print("{0}: not available".format(suffix))
            continue

        try:
            shape = read_nifti_shape(file_path)
            print("{0}: {1}".format(suffix, shape))
        except Exception as exc:
            print("{0}: failed to read ({1})".format(suffix, exc))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read all patient folders and print image sizes for 4 images and 1 segmentation."
    )
    parser.add_argument(
        "--path-profile",
        choices=["local", "legacy"],
        default="local",
        help="local uses ../archive next to the repository; legacy keeps the original F: drive path.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Root directory that contains patient subfolders.",
    )
    args = parser.parse_args()
    root = Path(args.root) if args.root is not None else (LEGACY_DATA_ROOT if args.path_profile == "legacy" else LOCAL_DATA_ROOT)

    if not os.path.isdir(root):
        raise FileNotFoundError("Directory does not exist: {0}".format(root))

    print(f"Path profile: {args.path_profile}")
    print(f"Data root   : {root.resolve()}")
    print_results(str(root))


if __name__ == "__main__":
    main()
