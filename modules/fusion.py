"""Weighted ensemble fusion of all original forensic signals into one
score. Every signal here is a self-implemented, explainable algorithm —
there is no black-box pretrained classifier anywhere in this pipeline.

If a signal is unavailable (e.g. the image is too small for a
patch-based analysis), its weight is redistributed proportionally
across the remaining available signals rather than defaulting to zero.
"""

DEFAULT_WEIGHTS = {
    "ela": 0.10,
    "frequency": 0.09,
    "noise": 0.13,
    "benford": 0.12,
    "double_compression": 0.07,
    "texture": 0.09,
    "chromatic_aberration": 0.10,
    "cfa": 0.10,
    "copy_move": 0.12,
    "metadata": 0.08,
}

LABELS = {
    "ela": "Error Level Analysis",
    "frequency": "Frequency-Domain (FFT) Analysis",
    "noise": "PRNU Sensor Noise Consistency",
    "benford": "Benford's Law (DCT Coefficients)",
    "double_compression": "Double-Compression Periodicity",
    "texture": "Texture Regularity (LBP Entropy)",
    "chromatic_aberration": "Chromatic Aberration Consistency",
    "cfa": "CFA / Demosaicing Footprint",
    "copy_move": "Copy-Move Forgery Detection",
    "metadata": "Metadata / EXIF Risk",
}


def metadata_score(metadata: dict) -> float:
    if not metadata["has_exif"]:
        return 60.0
    score = 20.0 * min(len(metadata["risk_flags"]), 3)
    return float(min(score, 100.0))


def fuse_scores(scores: dict, weights: dict = None) -> dict:
    """scores: dict of component_name -> 0..100 score, or None if unavailable."""
    weights = weights or DEFAULT_WEIGHTS
    available = {k: v for k, v in scores.items() if v is not None}
    total_weight = sum(weights[k] for k in available)
    if total_weight == 0:
        return {"final_score": 0.0, "components": {}, "weights": {}, "verdict": classify(0.0)}

    final = sum(available[k] * weights[k] for k in available) / total_weight
    normalized_weights = {k: weights[k] / total_weight for k in available}

    return {
        "final_score": float(final),
        "components": available,
        "weights": normalized_weights,
        "verdict": classify(final),
    }


def classify(score: float) -> dict:
    if score < 35:
        return {"label": "LIKELY AUTHENTIC", "color": "#00FFB3", "badge": "bg"}
    elif score < 65:
        return {"label": "INCONCLUSIVE — REVIEW REQUIRED", "color": "#FFD700", "badge": "ba"}
    else:
        return {"label": "LIKELY MANIPULATED OR AI-GENERATED", "color": "#FF3366", "badge": "br"}
