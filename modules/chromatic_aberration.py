"""Original lateral chromatic-aberration consistency analysis — built
from scratch with patch-wise cross-correlation, no pretrained models.

Real camera lenses bend different wavelengths of light by slightly
different amounts, producing a small, physically consistent radial
misalignment between colour channels that grows from the image centre
toward the edges. Purely synthetic imagery — rendered without a lens
model — usually shows negligible or spatially inconsistent channel
misalignment instead of this radial pattern.
"""
import numpy as np
from PIL import Image


def _best_shift(ref: np.ndarray, target: np.ndarray, search: int = 2):
    best_score, best_shift = -np.inf, (0, 0)
    ref_c = ref - ref.mean()
    ref_norm = np.linalg.norm(ref_c) + 1e-8
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            shifted = np.roll(np.roll(target, dy, axis=0), dx, axis=1)
            shifted_c = shifted - shifted.mean()
            denom = ref_norm * (np.linalg.norm(shifted_c) + 1e-8)
            score = float(np.sum(ref_c * shifted_c) / denom)
            if score > best_score:
                best_score, best_shift = score, (dy, dx)
    return best_shift


def chromatic_aberration_analysis(image: Image.Image, grid: int = 5, patch: int = 48) -> dict:
    rgb = np.array(image.convert("RGB")).astype(np.float32)
    h, w, _ = rgb.shape

    if h < patch * 2 or w < patch * 2:
        return {"score": 0.0, "correlation": 0.0, "mean_magnitude": 0.0,
                "insufficient_data": True, "samples": []}

    cy, cx = h / 2.0, w / 2.0
    half = patch // 2
    ys = np.linspace(half, h - half, grid).astype(int)
    xs = np.linspace(half, w - half, grid).astype(int)

    radii, magnitudes = [], []
    for y in ys:
        for x in xs:
            r_patch = rgb[y - half:y + half, x - half:x + half, 0]
            g_patch = rgb[y - half:y + half, x - half:x + half, 1]
            b_patch = rgb[y - half:y + half, x - half:x + half, 2]

            dy_r, dx_r = _best_shift(g_patch, r_patch)
            dy_b, dx_b = _best_shift(g_patch, b_patch)

            mag = float(np.hypot(dy_r, dx_r) + np.hypot(dy_b, dx_b))
            radius = float(np.hypot(y - cy, x - cx))
            radii.append(radius)
            magnitudes.append(mag)

    radii_arr = np.array(radii)
    mags_arr = np.array(magnitudes)

    if mags_arr.std() < 1e-6 or radii_arr.std() < 1e-6:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(radii_arr, mags_arr)[0, 1])
        if np.isnan(correlation):
            correlation = 0.0

    mean_mag = float(mags_arr.mean())
    if mean_mag < 0.15:
        # No measurable channel misalignment anywhere in the frame.
        score = 70.0
    else:
        score = float(np.clip((0.35 - max(correlation, -1.0)) * 130, 0, 100))

    # normalize radii to 0-1 for plotting convenience
    max_r = float(radii_arr.max()) or 1.0
    samples = [{"radius": r / max_r, "magnitude": m} for r, m in zip(radii, magnitudes)]

    return {
        "score": score,
        "correlation": correlation,
        "mean_magnitude": mean_mag,
        "insufficient_data": False,
        "samples": samples,
    }
