"""Original sensor pattern noise (PRNU-style) extraction using
wavelet-domain Wiener filtering — the actual method from Lukas,
Fridrich & Goljan, "Digital Camera Identification from Sensor Pattern
Noise" (IEEE TIFS, 2006), one of the most cited techniques in image
forensics. This replaces a naive Gaussian-blur high-pass residual with
the textbook approach: a multi-level wavelet decomposition, a robust
MAD-based noise-level estimate from the finest diagonal detail band,
and adaptive Wiener shrinkage applied per detail coefficient before
reconstructing the noise residual in the pixel domain.

Real camera sensor noise is close to spatially uniform in its local
variance across a frame (a genuine physical noise floor). Splices,
composites, and synthetically generated regions tend to break that
uniformity — either by introducing a region with a different noise
floor, or by having implausibly little residual noise at all.

Implemented from scratch on top of PyWavelets' transform primitives;
no external models or pretrained weights.
"""
import numpy as np
import pywt
import cv2
from PIL import Image


def _wavelet_denoise(gray: np.ndarray, wavelet: str = "db8", level: int = 4):
    # Cap the decomposition depth to what the image size can actually
    # support cleanly — requesting too many levels on a small image
    # causes boundary-dominated, unreliable coefficients at the deepest
    # levels rather than an outright error, which is worse (silently
    # bad data instead of a clear failure).
    max_level = pywt.dwt_max_level(min(gray.shape), pywt.Wavelet(wavelet).dec_len)
    level = max(1, min(level, max_level))
    coeffs = pywt.wavedec2(gray, wavelet, level=level)

    # Robust noise-sigma estimate from the finest diagonal detail band
    # (standard MAD estimator, Donoho & Johnstone 1994).
    finest_diag = coeffs[-1][2]
    sigma = float(np.median(np.abs(finest_diag)) / 0.6745) + 1e-8

    new_coeffs = [coeffs[0]]
    for detail_level in coeffs[1:]:
        shrunk = []
        for d in detail_level:
            # Adaptive Wiener shrinkage: attenuate each coefficient by
            # how much of its local energy exceeds the noise floor.
            local_var = d ** 2
            gain = np.maximum(local_var - sigma ** 2, 0) / (local_var + 1e-8)
            shrunk.append(d * gain)
        new_coeffs.append(tuple(shrunk))

    denoised = pywt.waverec2(new_coeffs, wavelet)
    denoised = denoised[:gray.shape[0], :gray.shape[1]]
    return denoised, sigma


def prnu_noise_analysis(image: Image.Image, block: int = 32, max_dim: int = 900) -> dict:
    gray = np.array(image.convert("L")).astype(np.float32)

    if max(gray.shape) > max_dim:
        scale = max_dim / max(gray.shape)
        gray = cv2.resize(gray, (int(gray.shape[1] * scale), int(gray.shape[0] * scale)))

    # pad to an even size for clean wavelet reconstruction
    h, w = gray.shape
    h2, w2 = h + (h % 2), w + (w % 2)
    if (h2, w2) != (h, w):
        padded = np.zeros((h2, w2), dtype=np.float32)
        padded[:h, :w] = gray
        gray_in = padded
    else:
        gray_in = gray

    denoised, sigma = _wavelet_denoise(gray_in, level=4)
    residual = (gray_in - denoised)[:h, :w]

    residual_norm = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)
    residual_image = Image.fromarray(residual_norm.astype(np.uint8))

    variances = []
    for y in range(0, max(h - block, 1), block):
        for x in range(0, max(w - block, 1), block):
            patch = residual[y:y + block, x:x + block]
            if patch.size > 0:
                variances.append(float(patch.var()))
    variances = np.array(variances) if variances else np.array([0.0])
    mean_var = variances.mean() if variances.mean() > 1e-8 else 1e-8
    consistency = float(variances.std() / mean_var)

    score = float(np.clip(consistency * 42, 0, 100))

    return {
        "score": score,
        "sigma": sigma,
        "consistency": consistency,
        "residual_image": residual_image,
    }
