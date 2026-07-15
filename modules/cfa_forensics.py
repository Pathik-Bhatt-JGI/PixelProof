"""Original Color Filter Array (CFA) / demosaicing-artifact analysis.

Real camera sensors capture colour through a periodic filter mosaic
(almost universally a Bayer pattern) and reconstruct full-resolution
channels via demosaicing interpolation. That interpolation leaves a
characteristic periodic correlation footprint at half the sampling
frequency (the Nyquist rate of the 2x2 CFA period) in a simple
neighbour-prediction residual of the green channel — green is sampled
at every other pixel in a Bayer array, so it carries the strongest,
most detectable footprint of the four common channel arrangements.

Purely synthetic imagery, screenshots, and heavily resampled images
generally lack this specific spectral footprint. This is a simplified,
from-scratch proxy for the periodicity-detection spirit of Popescu &
Farid's EM-based CFA analysis (2005) — implemented here directly from
pixel arithmetic and a 2D FFT, no external models.
"""
import numpy as np
from PIL import Image


def _local_max(arr: np.ndarray, y: int, x: int, r: int = 1) -> float:
    y0, y1 = max(y - r, 0), min(y + r + 1, arr.shape[0])
    x0, x1 = max(x - r, 0), min(x + r + 1, arr.shape[1])
    return float(arr[y0:y1, x0:x1].max())


def cfa_analysis(image: Image.Image) -> dict:
    rgb = np.array(image.convert("RGB")).astype(np.float32)
    green = rgb[:, :, 1]

    if max(green.shape) > 700:
        scale = 700 / max(green.shape)
        new_size = (int(green.shape[1] * scale), int(green.shape[0] * scale))
        green = np.array(Image.fromarray(green.astype(np.uint8)).resize(new_size)).astype(np.float32)

    h, w = green.shape
    if h < 32 or w < 32:
        return {"score": 0.0, "insufficient_data": True, "periodicity_index": 0.0, "residual_image": None}

    # simple 4-neighbour average predictor — demosaicing-interpolated
    # pixels are, by construction, close to a local average of their
    # neighbours; directly-sampled pixels are not.
    pred = np.zeros_like(green)
    pred[1:-1, 1:-1] = (green[:-2, 1:-1] + green[2:, 1:-1] + green[1:-1, :-2] + green[1:-1, 2:]) / 4.0
    residual = (green - pred)[1:-1, 1:-1]

    f = np.fft.fft2(residual - residual.mean())
    fshift = np.abs(np.fft.fftshift(f))
    hh, ww = fshift.shape
    cy, cx = hh // 2, ww // 2

    # After fftshift, index 0 on each axis corresponds to that axis's
    # Nyquist frequency (period-2 checkerboard structure); (0,0) is the
    # diagonal Nyquist point. A camera-typical CFA footprint shows up as
    # a sharp peak at one or more of these three points.
    v_peak = _local_max(fshift, 0, cx)
    h_peak = _local_max(fshift, cy, 0)
    d_peak = _local_max(fshift, 0, 0)
    peak = max(v_peak, h_peak, d_peak)

    baseline = float(np.median(fshift)) + 1e-6
    periodicity_index = float(peak / baseline)

    # A strong Nyquist-frequency peak is the camera-typical demosaicing
    # footprint; its absence raises the "no CFA footprint detected"
    # score, since that footprint is difficult to fake without actually
    # rendering through a real Bayer pipeline.
    score = float(np.clip((9.0 - periodicity_index) * 7.0, 0, 100))

    residual_vis = np.clip(np.abs(residual) * 4, 0, 255).astype(np.uint8)

    return {
        "score": score,
        "insufficient_data": False,
        "periodicity_index": periodicity_index,
        "residual_image": Image.fromarray(residual_vis),
    }
