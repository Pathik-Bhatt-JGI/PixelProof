"""Converts the outputs of ForensiQ's ten original forensic modules into
a fixed-order numeric feature vector suitable for training a classifier.

This is the bridge between the hand-engineered signal-processing layer
(everything in modules/forensics.py, prnu.py, dct_forensics.py, etc.)
and a properly *calibrated* classifier trained on real labeled data —
rather than the hand-picked heuristic thresholds those modules use on
their own. The features below are richer than just the ten 0-100
scores: they include the underlying statistics each score is derived
from, giving a downstream classifier more to learn from than the
already-compressed heuristic output.

Every value here traces back to a self-implemented signal — nothing in
this feature vector comes from a pretrained model.
"""
from PIL import Image

from . import forensics, prnu, dct_forensics, texture_forensics, chromatic_aberration
from . import cfa_forensics, copy_move, metadata as meta_mod, fusion

# Fixed feature order — must stay stable between training and inference.
FEATURE_NAMES = [
    "ela_score", "ela_mean_error", "ela_std_error",
    "freq_score", "freq_high_low_ratio",
    "noise_score", "noise_sigma", "noise_consistency",
    "benford_score", "benford_chi_square",
    "dcomp_score", "dcomp_periodicity_ratio",
    "texture_score", "texture_entropy_ratio", "texture_uniform_fraction",
    "ca_score", "ca_correlation", "ca_mean_magnitude",
    "cfa_score", "cfa_periodicity_index",
    "copy_move_score", "copy_move_match_ratio",
    "metadata_score", "metadata_has_exif", "metadata_num_flags",
]


def extract_feature_vector(image: Image.Image) -> dict:
    """Returns a dict of feature_name -> float. Any signal that reports
    insufficient_data falls back to a neutral placeholder (0.0) with a
    companion `<signal>_available` flag left out of the vector for
    simplicity — callers doing large-scale extraction should filter or
    impute as appropriate for their dataset.
    """
    ela = forensics.error_level_analysis(image)
    freq = forensics.frequency_analysis(image)
    noise = prnu.prnu_noise_analysis(image)
    benford = dct_forensics.benford_analysis(image)
    dcomp = dct_forensics.double_compression_analysis(image)
    texture = texture_forensics.texture_regularity_analysis(image)
    ca = chromatic_aberration.chromatic_aberration_analysis(image)
    cfa = cfa_forensics.cfa_analysis(image)
    cm = copy_move.copy_move_analysis(image)
    md = meta_mod.extract_metadata(image)

    return feats_from_results(ela, freq, noise, benford, dcomp, texture, ca, cfa, cm, md)


def feats_from_results(ela: dict, freq: dict, noise: dict, benford: dict, dcomp: dict,
                        texture: dict, ca: dict, cfa: dict, cm: dict, md: dict) -> dict:
    """Same feature set as extract_feature_vector, but built from result
    dicts the caller already computed (app.py computes each signal once
    for its own tabs/visuals; this avoids running every module twice).
    """
    return {
        "ela_score": ela["score"],
        "ela_mean_error": ela["mean_error"],
        "ela_std_error": ela["std_error"],
        "freq_score": freq["score"],
        "freq_high_low_ratio": freq["high_low_ratio"],
        "noise_score": noise["score"],
        "noise_sigma": noise["sigma"],
        "noise_consistency": noise["consistency"],
        "benford_score": 0.0 if benford["insufficient_data"] else benford["score"],
        "benford_chi_square": 0.0 if benford["insufficient_data"] else benford["chi_square"],
        "dcomp_score": 0.0 if dcomp["insufficient_data"] else dcomp["score"],
        "dcomp_periodicity_ratio": 0.0 if dcomp["insufficient_data"] else dcomp["periodicity_ratio"],
        "texture_score": texture["score"],
        "texture_entropy_ratio": texture["entropy_ratio"],
        "texture_uniform_fraction": texture["uniform_fraction"],
        "ca_score": 0.0 if ca["insufficient_data"] else ca["score"],
        "ca_correlation": 0.0 if ca["insufficient_data"] else ca["correlation"],
        "ca_mean_magnitude": 0.0 if ca["insufficient_data"] else ca["mean_magnitude"],
        "cfa_score": 0.0 if cfa["insufficient_data"] else cfa["score"],
        "cfa_periodicity_index": 0.0 if cfa["insufficient_data"] else cfa["periodicity_index"],
        "copy_move_score": 0.0 if cm["insufficient_data"] else cm["score"],
        "copy_move_match_ratio": 0.0 if cm["insufficient_data"] or cm["total_blocks"] == 0
                                  else cm["match_count"] / cm["total_blocks"],
        "metadata_score": fusion.metadata_score(md),
        "metadata_has_exif": 1.0 if md["has_exif"] else 0.0,
        "metadata_num_flags": float(len(md["risk_flags"])),
    }


def feature_dict_to_vector(feats: dict) -> list:
    return [feats[name] for name in FEATURE_NAMES]
