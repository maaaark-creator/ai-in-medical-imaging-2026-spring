from __future__ import annotations

from pathlib import Path

from PIL import Image


def crop_box(image: Image.Image, left: float, top: float, right: float, bottom: float) -> Image.Image:
    w, h = image.size
    return image.crop((int(w * left), int(h * top), int(w * right), int(h * bottom)))


def main() -> None:
    out_dir = Path("outputs") / "fusion_figure_assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    triptych = Image.open("outputs/task1_vertical_line_assets/triptychs/BraTS-GLI-00000-000_slice_077_triptych.png")
    pair_error = Image.open("outputs/task1_vertical_line_assets/pair_error/BraTS-GLI-00000-000_slice_077_pair_error.png")
    kspace_full = Image.open("outputs/kspace_previews/BraTS-GLI-00000-000-t2w_slice_077_kspace_preview.png")
    kspace_masked = Image.open("outputs/masked_kspace_previews/BraTS-GLI-00000-000_slice_077_masked_kspace.png")

    full_img = crop_box(triptych, 0.35, 0.16, 0.63, 0.84)
    alias_img = crop_box(triptych, 0.67, 0.16, 0.95, 0.84)
    gt_img = crop_box(pair_error, 0.03, 0.13, 0.31, 0.88)

    k2_img = crop_box(kspace_masked, 0.08, 0.18, 0.34, 0.83)
    k1_img = crop_box(kspace_full, 0.12, 0.17, 0.38, 0.84)
    kout_img = crop_box(kspace_masked, 0.65, 0.15, 0.92, 0.85)

    assets = {
        "t2_input.png": alias_img,
        "t1_input.png": gt_img,
        "gt_image.png": full_img,
        "image_output.png": alias_img,
        "k2_input.png": k2_img,
        "k1_input.png": k1_img,
        "gt_kspace.png": k1_img,
        "kspace_output.png": kout_img,
    }

    for name, image in assets.items():
        image.resize((160, 160), Image.Resampling.LANCZOS).save(out_dir / name)

    print(out_dir.resolve())


if __name__ == "__main__":
    main()
