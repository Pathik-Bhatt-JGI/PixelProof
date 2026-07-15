"""Original DCT-domain forensic analysis — built from first principles,
no pretrained models or external weights.

Two established digital-forensics techniques, implemented here from
scratch on top of the raw discrete cosine transform:

1. Benford's Law conformity: natural photographs — after passing through
   sensor noise, lens optics, and real-world JPEG quantization — have
   DCT AC coefficients whose leading digits closely follow Benford's
   Law. Synthetic or heavily reprocessed imagery tends to diverge from
   it (Fu, Shi & Su, 2007).
2. Double-compression periodicity: a second JPEG compression pass at a
   different quality factor imprints a periodic, comb-like structure on
   the histogram of a DCT coefficient (Popescu & Farid, 2004).
"""
import numpy as np
import cv2


def _block_dct_coefficients(gray: np.ndarray, positions):
    h, w = gray.shape
    h8, w8 = h - h % 8, w - w % 8
    gray = gray[:h8, :w8].astype(np.float32)

    coeffs = {p: [] for p in positions}
    for y in range(0, h8, 8):
        for x in range(0, w8, 8):
            block = gray[y:y + 8, x:x + 8]
            d = cv2.dct(block)
            for (i, j) in positions:
                coeffs[(i, j)].append(float(d[i, j]))
    return coeffs


def _leading_digits(values):
    digits = []
    for v in values:
        av = abs(v)
        if av < 1e-3:
            continue
        while av < 1:
            av *= 10
        while av >= 10:
            av /= 10
        digits.append(int(av))
    return digits


BENFORD_REF = np.array([np.log10(1 + 1.0 / d) for d in range(1, 10)])
DCT_POSITIONS = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1)]


def benford_analysis(image) -> dict:
    gray = np.array(image.convert("L"))
    coeffs = _block_dct_coefficients(gray, DCT_POSITIONS)
    all_values = [v for p in DCT_POSITIONS for v in coeffs[p]]
    digits = _leading_digits(all_values)

    if len(digits) < 200:
        return {"score": 0.0, "chi_square": 0.0, "observed": None,
                "expected": BENFORD_REF.tolist(), "insufficient_data": True}

    counts = np.bincount(digits, minlength=10)[1:10].astype(np.float64)
    observed = counts / counts.sum()
    expected_counts = BENFORD_REF * counts.sum()
    chi_square = float(np.sum((counts - expected_counts) ** 2 / (expected_counts + 1e-8)))

    # Scaled heuristically: low chi-square (close fit) -> low score,
    # rising divergence -> rising score, capped at 100.
    score = float(np.clip(chi_square * 1.1, 0, 100))

    return {
        "score": score,
        "chi_square": chi_square,
        "observed": observed.tolist(),
        "expected": BENFORD_REF.tolist(),
        "insufficient_data": False,
    }


def double_compression_analysis(image) -> dict:
    gray = np.array(image.convert("L"))
    coeffs = _block_dct_coefficients(gray, [(1, 1)])[(1, 1)]
    values = np.round(np.array(coeffs)).astype(np.int32)

    if len(values) < 200:
        return {"score": 0.0, "periodicity_ratio": 0.0, "histogram": None,
                "bin_edges": None, "insufficient_data": True}

    lo, hi = int(np.percentile(values, 1)), int(np.percentile(values, 99))
    hi = max(hi, lo + 8)
    hist, edges = np.histogram(values, bins=np.arange(lo, hi + 2))
    hist = hist.astype(np.float64)

    spectrum = np.abs(np.fft.rfft(hist - hist.mean()))
    if len(spectrum) > 3:
        ac_spectrum = spectrum[1:]
        peak = float(ac_spectrum.max())
        baseline = float(np.median(ac_spectrum)) + 1e-8
        periodicity_ratio = peak / baseline
    else:
        periodicity_ratio = 0.0

    score = float(np.clip((periodicity_ratio - 3.0) * 11, 0, 100))

    return {
        "score": score,
        "periodicity_ratio": periodicity_ratio,
        "histogram": hist.tolist(),
        "bin_edges": edges.tolist(),
        "insufficient_data": False,
    }
