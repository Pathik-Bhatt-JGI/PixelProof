"""Unit tests for ForensiQ's forensic modules.

Run with:  pytest tests/ -v

These aren't meant to validate forensic *accuracy* against a labeled
benchmark dataset (that requires curated ground-truth imagery this repo
doesn't ship with) — they validate that each algorithm behaves
correctly on cases with a mathematically known right answer: hash
determinism, weight redistribution arithmetic, a synthetically inserted
copy-move forgery with a known shift vector, a synthetically inserted
noise-floor splice, LBP uniform-pattern classification against the
textbook definition, and Benford digit extraction correctness.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import hashing, fusion, texture_forensics, dct_forensics, copy_move, prnu


# --------------------------------------------------------------- hashing --
def test_hash_deterministic():
    data = b"forensiq test evidence bytes"
    h1 = hashing.compute_hashes(data)
    h2 = hashing.compute_hashes(data)
    assert h1["sha256"] == h2["sha256"]
    assert h1["md5"] == h2["md5"]
    assert h1["size_bytes"] == len(data)


def test_hash_changes_with_content():
    h1 = hashing.compute_hashes(b"evidence A")
    h2 = hashing.compute_hashes(b"evidence B")
    assert h1["sha256"] != h2["sha256"]


# ---------------------------------------------------------------- fusion --
def test_fuse_scores_all_available_sums_weights_to_one():
    scores = {k: 50.0 for k in fusion.DEFAULT_WEIGHTS}
    result = fusion.fuse_scores(scores)
    assert abs(sum(result["weights"].values()) - 1.0) < 1e-9
    assert abs(result["final_score"] - 50.0) < 1e-9


def test_fuse_scores_redistributes_missing_signal_weight():
    scores = {k: 80.0 for k in fusion.DEFAULT_WEIGHTS}
    scores["copy_move"] = None  # simulate an unavailable signal
    result = fusion.fuse_scores(scores)
    assert "copy_move" not in result["weights"]
    assert abs(sum(result["weights"].values()) - 1.0) < 1e-9
    assert abs(result["final_score"] - 80.0) < 1e-6


def test_fuse_scores_no_signals_available():
    scores = {k: None for k in fusion.DEFAULT_WEIGHTS}
    result = fusion.fuse_scores(scores)
    assert result["final_score"] == 0.0
    assert result["components"] == {}


def test_classify_thresholds():
    assert fusion.classify(10)["label"] == "LIKELY AUTHENTIC"
    assert fusion.classify(50)["label"].startswith("INCONCLUSIVE")
    assert fusion.classify(90)["label"].startswith("LIKELY MANIPULATED")


# ----------------------------------------------------------- texture/LBP --
def test_lbp_uniform_pattern_matches_textbook_definition():
    # 0 transitions (all-zero code) is uniform
    assert texture_forensics._is_uniform(0b00000000) is True
    # exactly 2 transitions (one contiguous run of 1s) is uniform
    assert texture_forensics._is_uniform(0b00011100) is True
    # many alternating bits (multiple transitions) is non-uniform
    assert texture_forensics._is_uniform(0b01010101) is False


def test_texture_entropy_ratio_in_valid_range():
    img = Image.fromarray((np.random.rand(200, 200) * 255).astype(np.uint8))
    result = texture_forensics.texture_regularity_analysis(img)
    assert 0.0 <= result["entropy_ratio"] <= 1.0
    assert 0.0 <= result["score"] <= 100.0


def test_flat_image_flagged_as_unnaturally_regular():
    flat = Image.fromarray(np.full((200, 200), 128, dtype=np.uint8))
    result = texture_forensics.texture_regularity_analysis(flat)
    # zero local variation everywhere -> minimal LBP entropy -> high score
    assert result["score"] > 50.0


# ------------------------------------------------------------- benford --
def test_leading_digit_extraction():
    digits = dct_forensics._leading_digits([123.0, 0.045, 987.6, -56.0])
    assert digits == [1, 4, 9, 5]


def test_leading_digit_skips_near_zero_values():
    digits = dct_forensics._leading_digits([0.0, 1e-6, 42.0])
    assert digits == [4]


def test_benford_reference_distribution_sums_to_one():
    assert abs(dct_forensics.BENFORD_REF.sum() - 1.0) < 1e-6
    assert dct_forensics.BENFORD_REF[0] > dct_forensics.BENFORD_REF[-1]  # digit 1 more common than 9


# ----------------------------------------------------------- copy-move --
def _make_copy_move_image(size=400, patch=60, shift=200, seed=1):
    rng = np.random.RandomState(seed)
    base = rng.randint(60, 200, (size, size, 3)).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    texture = 40 * np.sin(yy / 25) + 30 * np.cos(xx / 18)
    base += texture[:, :, None]
    base = np.clip(base, 0, 255).astype(np.uint8)

    forged = base.copy()
    src = forged[50:50 + patch, 50:50 + patch].copy()
    forged[50 + shift:50 + shift + patch, 50 + shift:50 + shift + patch] = src
    return Image.fromarray(base), Image.fromarray(forged)


def test_copy_move_detects_known_shift_vector():
    _, forged = _make_copy_move_image()
    result = copy_move.copy_move_analysis(forged)
    assert not result["insufficient_data"]
    assert result["match_count"] > 0
    assert result["dominant_shift"] == (200, 200)
    assert result["score"] > 50.0


def test_copy_move_clean_image_scores_low():
    clean, _ = _make_copy_move_image()
    result = copy_move.copy_move_analysis(clean)
    assert result["score"] < 20.0


def test_copy_move_insufficient_data_on_tiny_image():
    tiny = Image.fromarray((np.random.rand(20, 20, 3) * 255).astype(np.uint8))
    result = copy_move.copy_move_analysis(tiny)
    assert result["insufficient_data"] is True


# ----------------------------------------------------------------- prnu --
def test_prnu_flags_noise_floor_splice():
    rng = np.random.RandomState(2)
    import cv2
    base = cv2.GaussianBlur(rng.randint(80, 180, (300, 300)).astype(np.float32), (0, 0), 3)
    noisy = np.clip(base + rng.normal(0, 12, (300, 300)), 0, 255).astype(np.uint8)

    spliced = noisy.copy().astype(np.float32)
    spliced[90:210, 90:210] = cv2.GaussianBlur(spliced[90:210, 90:210], (0, 0), 5)
    spliced = np.clip(spliced, 0, 255).astype(np.uint8)

    uniform_result = prnu.prnu_noise_analysis(Image.fromarray(noisy))
    spliced_result = prnu.prnu_noise_analysis(Image.fromarray(spliced))
    assert spliced_result["consistency"] > uniform_result["consistency"]


# ------------------------------------------------------ feature extraction --
def test_feature_extraction_returns_all_expected_keys():
    from modules import feature_extraction as fe
    img = Image.fromarray((np.random.rand(200, 200, 3) * 255).astype(np.uint8))
    feats = fe.extract_feature_vector(img)
    assert set(feats.keys()) == set(fe.FEATURE_NAMES)
    assert all(isinstance(v, (int, float)) for v in feats.values())


def test_feature_dict_to_vector_preserves_order():
    from modules import feature_extraction as fe
    feats = {name: float(i) for i, name in enumerate(fe.FEATURE_NAMES)}
    vec = fe.feature_dict_to_vector(feats)
    assert vec == [float(i) for i in range(len(fe.FEATURE_NAMES))]


# ---------------------------------------- optional trained-model fallback --
def test_learned_fusion_unavailable_on_fresh_checkout():
    from modules import learned_fusion
    # A fresh checkout of this repo ships no trained/ folder — the app
    # must degrade gracefully rather than error.
    if not learned_fusion.TRAINED_DIR.exists():
        assert learned_fusion.is_available() is False


def test_cnn_detector_unavailable_on_fresh_checkout():
    from modules import cnn_detector
    if not cnn_detector.TRAINED_CNN_DIR.exists():
        assert cnn_detector.is_available() is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
