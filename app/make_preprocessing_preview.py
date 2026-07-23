"""Generate side-by-side preview images for presentation slides:
   left = original MRI, right = after the inference preprocessing pipeline
   (CLAHE + resize 224x224). The ImageNet normalization step is invisible to
   the eye so it's not shown; the CLAHE step is what changes the look.

Outputs PNGs into reports/preprocessing_preview/.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT       = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "static" / "samples" / "brisc2025"
OUT_DIR    = ROOT.parent / "reports" / "preprocessing_preview"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# One representative image per class.
PICKS = [
    ("glioma",     "Glioma"),
    ("meningioma", "Meningioma"),
    ("no_tumor",   "No Tumor"),
    ("pituitary",  "Pituitary"),
]

# Display size of each panel in the side-by-side preview.
PANEL_SIDE = 512


def apply_inference_clahe(pil_img: Image.Image) -> Image.Image:
    """Mirror what app.py does inside _validate_and_enhance:
       always-on CLAHE with clipLimit=2.0, tileGridSize=8x8."""
    arr = np.array(pil_img.convert("RGB"))
    arr_uint8 = arr if arr.dtype == np.uint8 else (arr * 255).astype(np.uint8)
    lab = cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)


def label_image(panel: Image.Image, text: str, accent=(167, 139, 250)) -> Image.Image:
    """Add a centered label bar at the bottom of a panel."""
    bar_height = 44
    out = Image.new("RGB", (panel.width, panel.height + bar_height), (12, 12, 18))
    out.paste(panel, (0, 0))
    draw = ImageDraw.Draw(out)
    # Accent strip on top of the bar
    draw.rectangle((0, panel.height, panel.width, panel.height + 3), fill=accent)
    # Try to load a nicer font, fallback to default
    font = None
    for candidate in ("arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            font = ImageFont.truetype(candidate, 22)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(((panel.width - tw) // 2, panel.height + 6 + (bar_height - 6 - th) // 2 - 2),
              text, fill=(240, 240, 245), font=font)
    return out


def build_pair(src: Path, cls_label: str) -> Image.Image:
    original = Image.open(src).convert("RGB")
    processed = apply_inference_clahe(original)

    # Resize both to the same square panel size for a fair visual comparison.
    # We use bicubic to keep edges crisp.
    orig_panel = original.resize((PANEL_SIDE, PANEL_SIDE), Image.BICUBIC)
    proc_panel = processed.resize((PANEL_SIDE, PANEL_SIDE), Image.BICUBIC)

    left  = label_image(orig_panel, "Original",   accent=(150, 150, 160))
    right = label_image(proc_panel, "Procesada (CLAHE + 224x224)", accent=(167, 139, 250))

    # Compose the two panels side by side with a thin separator.
    gap = 6
    total_w = left.width + gap + right.width
    canvas = Image.new("RGB", (total_w, left.height + 56), (12, 12, 18))
    canvas.paste(left,  (0, 56))
    canvas.paste(right, (left.width + gap, 56))

    # Title bar across the top.
    draw = ImageDraw.Draw(canvas)
    title_font = None
    for candidate in ("arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            title_font = ImageFont.truetype(candidate, 26)
            break
        except (OSError, IOError):
            continue
    if title_font is None:
        title_font = ImageFont.load_default()
    title = f"Preprocesado de imagen — {cls_label}"
    bbox = draw.textbbox((0, 0), title, font=title_font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(((total_w - tw) // 2, (56 - th) // 2 - 2),
              title, fill=(220, 220, 230), font=title_font)
    return canvas


def main() -> None:
    print(f"output dir: {OUT_DIR}")
    for folder, cls_label in PICKS:
        src_dir = SAMPLE_DIR / folder
        if not src_dir.exists():
            print(f"  (missing) {src_dir}")
            continue
        src = next(p for p in sorted(src_dir.iterdir()) if p.is_file())
        out = build_pair(src, cls_label)
        out_path = OUT_DIR / f"preprocessing_{folder}.png"
        out.save(out_path, format="PNG")
        print(f"  saved {out_path.name}  ({src.name})")


if __name__ == "__main__":
    main()
