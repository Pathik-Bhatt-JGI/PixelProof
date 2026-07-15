"""Composite manipulation-localization overlay.

Blends the Error Level Analysis map and the PRNU noise-residual map
into a single spatial heat overlay on top of the original image, so an
examiner can see *where* the strongest anomaly signals concentrate
rather than only a single global number. This introduces no new
detection logic of its own — it's a visualization layer over signals
already computed elsewhere in the pipeline, which is exactly how real
forensic report tooling presents multi-signal evidence.
"""
import numpy as np
from PIL import Image


def _to_norm_array(img: Image.Image, size) -> np.ndarray:
    arr = np.array(img.convert("L").resize(size)).astype(np.float32)
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def build_localization_overlay(original: Image.Image, ela_image: Image.Image,
                                noise_image: Image.Image, alpha: float = 0.42) -> Image.Image:
    size = original.size
    base = np.array(original.convert("RGB").resize(size)).astype(np.float32)

    ela_map = _to_norm_array(ela_image, size)
    noise_map = _to_norm_array(noise_image, size)
    combined = np.clip(0.6 * ela_map + 0.4 * noise_map, 0, 1)

    heat = np.zeros((size[1], size[0], 3), dtype=np.float32)
    heat[..., 0] = np.clip(combined * 3 - 1, 0, 1) * 255          # red ramps up at high suspicion
    heat[..., 1] = np.clip(1.2 - np.abs(combined - 0.4) * 2, 0, 1) * 255  # green/cyan mid-band
    heat[..., 2] = np.clip(1 - combined * 2, 0, 1) * 120           # subtle blue at low end

    blended = base * (1 - alpha) + heat * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
