"""Original texture-regularity forensic analysis using a from-scratch
Local Binary Pattern (LBP) implementation — no external CV/model
libraries, just numpy array shifts.

Real camera micro-texture (sensor noise, optical grain, fine surface
detail) produces a rich, high-entropy spread across LBP codes.
Over-smoothed or up-sampled synthetic regions common in AI image
generation often collapse that spread into a narrower band of patterns.
This module measures where an image's LBP-code entropy falls relative
to a typical natural-photo range and flags images that sit noticeably
outside it in either direction.
"""
import numpy as np
from PIL import Image


def _compute_lbp(gray: np.ndarray) -> np.ndarray:
    gray = gray.astype(np.int16)
    h, w = gray.shape
    padded = np.pad(gray, 1, mode="edge")
    center = padded[1:h + 1, 1:w + 1]

    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    code = np.zeros((h, w), dtype=np.uint8)
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        code = code | (((neighbor >= center).astype(np.uint8)) << bit)
    return code


def _is_uniform(code: int) -> bool:
    bits = f"{code:08b}"
    circular = bits + bits[0]
    transitions = sum(circular[i] != circular[i + 1] for i in range(8))
    return transitions <= 2


_UNIFORM_LUT = np.array([_is_uniform(i) for i in range(256)])


def texture_regularity_analysis(image: Image.Image) -> dict:
    gray = np.array(image.convert("L"))

    if max(gray.shape) > 900:
        scale = 900 / max(gray.shape)
        new_size = (int(gray.shape[1] * scale), int(gray.shape[0] * scale))
        gray = np.array(Image.fromarray(gray).resize(new_size))

    codes = _compute_lbp(gray)
    hist = np.bincount(codes.ravel(), minlength=256).astype(np.float64)
    hist_norm = hist / hist.sum()

    nz = hist_norm[hist_norm > 0]
    entropy = float(-np.sum(nz * np.log2(nz)))
    entropy_ratio = entropy / np.log2(256)
    uniform_fraction = float(hist_norm[_UNIFORM_LUT].sum())

    # Reference band from typical natural-photo sensor texture. Values
    # noticeably outside this band — either unnaturally flat/regular or
    # unnaturally chaotic — are flagged as anomalous.
    LOW, HIGH = 0.62, 0.80
    if entropy_ratio < LOW:
        deviation = (LOW - entropy_ratio) / LOW
    elif entropy_ratio > HIGH:
        deviation = (entropy_ratio - HIGH) / (1 - HIGH)
    else:
        deviation = 0.0

    score = float(np.clip(deviation * 140, 0, 100))
    lbp_image = Image.fromarray(codes.astype(np.uint8))

    return {
        "score": score,
        "entropy_ratio": entropy_ratio,
        "uniform_fraction": uniform_fraction,
        "lbp_image": lbp_image,
    }
