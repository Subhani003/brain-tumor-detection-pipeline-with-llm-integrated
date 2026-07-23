"""
Cross-scanner preprocessing helpers for inference.

Pipeline (applied to every image before the CNN sees it):
  1. brain_extract_and_crop  — Otsu mask → remove skull/background →
                               crop to brain bounding box with 10% padding.
  2. nyul_normalize           — Piecewise linear histogram matching to the
                               standard landmarks computed on the combined
                               training set.

Both functions accept a PIL RGB image and return a PIL RGB image so they slot
directly into the existing _preprocess_pil() chain in app.py.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Lazy-load landmarks once on first call
_LANDMARKS: dict | None = None
_LANDMARKS_PATH = Path(__file__).parent.parent / 'models' / 'nyul_landmarks.pkl'


def _load_landmarks() -> dict | None:
    global _LANDMARKS
    if _LANDMARKS is not None:
        return _LANDMARKS
    if _LANDMARKS_PATH.exists():
        try:
            _LANDMARKS = pickle.loads(_LANDMARKS_PATH.read_bytes())
            print(f'[preprocessing] Nyúl landmarks loaded from {_LANDMARKS_PATH}', flush=True)
        except Exception as e:
            print(f'[preprocessing] WARN: could not load landmarks ({e})', flush=True)
    else:
        print(f'[preprocessing] WARN: landmarks not found at {_LANDMARKS_PATH}. '
              f'Run src/compute_nyul_landmarks.py first.', flush=True)
    return _LANDMARKS


def _get_brain_mask(gray: np.ndarray) -> np.ndarray:
    """Otsu threshold + morphological close → binary brain mask."""
    arr_u8 = gray.astype(np.uint8)
    _, mask = cv2.threshold(arr_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel  = np.ones((7, 7), np.uint8)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Keep only the largest connected component (the brain)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = ((labels == largest) * 255).astype(np.uint8)
    return mask


def brain_extract_and_crop(pil_img: Image.Image, padding: float = 0.08) -> Image.Image:
    """
    Remove background/skull noise and crop to the brain's bounding box.

    Helps when different clinics/scanners produce images with different amounts
    of padding, skull visibility, or zoom — the brain content always ends up
    occupying the full frame after cropping.
    """
    arr  = np.array(pil_img.convert('RGB'))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    mask = _get_brain_mask(gray)
    # Bounding box of brain region
    coords = cv2.findNonZero(mask)
    if coords is None:
        return pil_img   # no mask → return unchanged
    x, y, w, h = cv2.boundingRect(coords)
    H, W = arr.shape[:2]
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W, x + w + pad_x)
    y2 = min(H, y + h + pad_y)
    if (x2 - x1) < 32 or (y2 - y1) < 32:
        return pil_img   # crop is tiny → return unchanged
    cropped = arr[y1:y2, x1:x2]
    return Image.fromarray(cropped)


def detect_mri_sequence(pil_img: Image.Image) -> dict:
    """
    Heuristic detector for non-T1 MRI sequences (T2, FLAIR, DWI, etc.).

    Key insight (empirically calibrated on combined training set):
      - T1 images have HIGH coefficient of variation (CV = std/mean) within the
        brain region (0.32–0.48) due to strong contrast between white matter,
        gray matter, and dark CSF ventricles.
      - T2 / FLAIR / DWI images have LOW CV (< 0.25) — they appear more
        homogeneous within the brain because CSF and tissue have different
        relative contrasts, or the DWI signal is very uniform.

    Threshold: CV < 0.28 → likely non-T1 sequence.

    Returns dict with keys:
        'is_non_t1'   : bool
        'cv'          : float  (coefficient of variation, main signal)
        'mean_brain'  : float
        'warning_msg' : str | None
    """
    arr  = np.array(pil_img.convert('L'))
    mask = _get_brain_mask(arr)
    brain = arr[mask > 0].astype(np.float32)
    if brain.size < 200:
        return {'is_non_t1': False, 'cv': None, 'mean_brain': None, 'warning_msg': None}
    mean_i = float(brain.mean())
    std_i  = float(brain.std())
    cv = std_i / (mean_i + 1e-8)
    is_non_t1 = bool(cv < 0.28)
    msg = None
    if is_non_t1:
        msg = (
            f'Low tissue contrast detected (CV={cv:.3f}, threshold=0.28). '
            'This image may be T2-weighted, FLAIR, or DWI. '
            'This model was trained on T1 sequences only — predictions on '
            'T2/FLAIR/DWI scans are unreliable regardless of confidence score.'
        )
    return {
        'is_non_t1':   is_non_t1,
        'cv':          round(cv, 4),
        'mean_brain':  round(mean_i, 2),
        'warning_msg': msg,
    }


def nyul_normalize(pil_img: Image.Image) -> Image.Image:
    """
    Piecewise linear histogram matching to the training-set standard landmarks.

    For each pixel *inside* the brain mask, linearly interpolate from the image's
    own percentile landmarks to the standard landmarks. Background pixels stay 0.

    If landmarks file is missing (not yet computed), returns the image unchanged
    with a one-time warning.
    """
    lm = _load_landmarks()
    if lm is None:
        return pil_img  # landmarks not computed yet → pass through

    std_lm = np.array(lm['standard_landmarks'], dtype=np.float32)
    pcts   = lm['percentiles']

    arr   = np.array(pil_img.convert('RGB'))
    gray  = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mask  = _get_brain_mask(gray.astype(np.uint8))
    brain = gray[mask > 0]

    if brain.size < 50:
        return pil_img

    # Compute this image's landmarks within the brain region
    img_lm = np.percentile(brain, pcts).astype(np.float32)

    # Build lookup table via piecewise linear interpolation
    #   f(pixel) = std_lm value at the corresponding position in img_lm
    lut = np.arange(256, dtype=np.float32)
    lut_mapped = np.interp(lut, img_lm, std_lm)
    lut_mapped = np.clip(lut_mapped, 0, 255).astype(np.uint8)

    # Apply to each channel
    out = np.zeros_like(arr)
    for c in range(3):
        ch = arr[:, :, c]
        mapped = lut_mapped[ch]
        # Zero out background pixels so we don't shift them
        mapped[mask == 0] = 0
        out[:, :, c] = mapped

    return Image.fromarray(out)
