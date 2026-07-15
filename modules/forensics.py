"""Classical digital-forensic signal analysis: Error Level Analysis and
frequency-domain (FFT) analysis. These don't require any downloaded
model and run entirely offline.

(Sensor-noise / PRNU analysis lives in modules/prnu.py, using proper
wavelet-domain Wiener filtering rather than a naive high-pass residual.)
"""
import io
import numpy as np
import cv2
from PIL import Image, ImageChops


def error_level_analysis(image: Image.Image, quality: int = 90, scale_cap: float = 15.0) -> dict:
    """Error Level Analysis: re-compress the image at a known JPEG quality
    and measure the difference. Regions that were edited/composited after
    the original compression tend to show a different error level than
    the rest of the image.
    """
    rgb = image.convert("RGB")
    buffer = io.BytesIO()
    rgb.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)

    diff = ImageChops.difference(rgb, resaved)
    diff_np = np.array(diff).astype(np.float32)

    max_diff = float(diff_np.max()) or 1.0
    scale_factor = min(255.0 / max_diff, scale_cap)
    ela_np = np.clip(diff_np * scale_factor, 0, 255).astype(np.uint8)
    ela_image = Image.fromarray(ela_np)

    mean_error = float(diff_np.mean())
    std_error = float(diff_np.std())

    # Heuristic: natural single-generation JPEGs usually show low, fairly
    # uniform error. Higher variance suggests localized re-compression
    # (editing) or a source that never went through normal camera JPEG
    # encoding at all (many AI generators export flat, ELA-uniform PNGs
    # re-saved as JPEG, which shows up as unusually LOW variance instead).
    score = float(np.clip((std_error / 12.0) * 100, 0, 100))

    return {
        "ela_image": ela_image,
        "mean_error": mean_error,
        "std_error": std_error,
        "score": score,
    }


def frequency_analysis(image: Image.Image) -> dict:
    """FFT-based frequency analysis. GAN/diffusion generators commonly
    leave periodic high-frequency artifacts (upsampling checkerboard
    patterns) that differ from the smoother 1/f falloff of natural
    camera sensor images.
    """
    gray = np.array(image.convert("L")).astype(np.float32)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    mag_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    spectrum_image = Image.fromarray((mag_norm * 255).astype(np.uint8))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.int32)

    radial_sum = np.bincount(r.ravel(), magnitude.ravel())
    radial_count = np.bincount(r.ravel())
    radial_mean = radial_sum / (radial_count + 1e-8)

    max_r = max(min(cy, cx), 4)
    low_band = radial_mean[: max_r // 4].mean()
    high_band = radial_mean[3 * max_r // 4 : max_r].mean()
    ratio = float(high_band / (low_band + 1e-8))

    score = float(np.clip((ratio - 0.15) * 400, 0, 100))

    return {
        "spectrum_image": spectrum_image,
        "high_low_ratio": ratio,
        "score": score,
    }



