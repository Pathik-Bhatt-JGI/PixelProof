# ForensiQ — Technical Methodology

This document explains the mathematical/statistical basis of each of the
ten forensic signals, the specific assumption each one tests, and its
known failure modes. It exists so this system can be defended in a viva,
an interview, or a technical review — not just demoed.

Every technique below is implemented directly from its mathematical
definition (numpy/opencv/scipy/PyWavelets primitives). None of them call
a pretrained classifier or external API.

---

## 1. Error Level Analysis (ELA)
**File:** `modules/forensics.py::error_level_analysis`

Re-saves the image at a known JPEG quality and measures the pixel-wise
difference from the original. Regions that were edited or composited
*after* the image's original compression pass tend to sit at a
different local error level than the rest of the frame, because they
haven't "settled" into the same quantization pattern.

**Fails on:** already-heavily-compressed images (little error left to
measure anywhere), and non-JPEG-original sources like clean PNG
screenshots or renders (which show near-uniform low error everywhere —
handled here as a low/flat signal rather than a false positive).

## 2. Frequency-Domain (FFT) Analysis
**File:** `modules/forensics.py::frequency_analysis`

Takes the 2D Fourier transform of the luminance channel and compares
radially-averaged energy in the high-frequency band to the low-frequency
band. Natural camera images follow roughly a 1/f falloff; GAN/diffusion
upsampling architectures commonly leave excess, unnaturally regular
high-frequency energy from their transposed-convolution/upsampling
stages.

**Fails on:** genuinely high-frequency-rich natural scenes (foliage,
fabric, gravel), which is why this is one signal among ten, not a
standalone verdict.

## 3. PRNU Sensor Noise Consistency
**File:** `modules/prnu.py::prnu_noise_analysis`

**Citation:** Lukas, Fridrich & Goljan, *"Digital Camera Identification
from Sensor Pattern Noise,"* IEEE Trans. Information Forensics and
Security, 2006.

A 4-level wavelet decomposition (Daubechies-8) is denoised with adaptive
Wiener shrinkage per detail coefficient, using a robust MAD-based
noise-sigma estimate from the finest diagonal detail band (Donoho &
Johnstone, 1994). The residual between the original and denoised image
approximates the sensor's physical pattern noise. Real camera noise is
close to spatially uniform in local variance across a frame; splices,
composites, and synthetic regions frequently break that uniformity —
either by having a different noise floor, or implausibly little noise
at all.

**Fails on:** very smooth/plain source scenes (sky, studio backdrops)
which naturally have low noise everywhere, and heavy in-camera noise
reduction that already flattens the sensor's noise signature before it
reaches this analysis.

## 4. Benford's Law on DCT Coefficients
**File:** `modules/dct_forensics.py::benford_analysis`

**Citation:** Fu, Shi & Su, *"A Generalized Benford's Law for JPEG
Coefficients and Its Applications in Image Forensics,"* SPIE Electronic
Imaging, 2007.

The leading (most-significant) digit of low/mid-frequency 8×8 block DCT
AC coefficients is extracted across the image and compared, via
chi-square divergence, to the classic Benford distribution
P(d) = log₁₀(1 + 1/d). Natural photographs — after sensor noise, lens
optics, and real JPEG quantization — closely follow this distribution;
synthetic or heavily reprocessed imagery tends to diverge.

**Fails on:** very small or very low-detail images (too few DCT blocks
for a stable digit distribution) — the module explicitly reports
`insufficient_data` and is excluded from fusion rather than guessed at.

## 5. Double-Compression Periodicity
**File:** `modules/dct_forensics.py::double_compression_analysis`

**Citation:** Popescu & Farid, *"Statistical Tools for Digital
Forensics,"* 6th International Workshop on Information Hiding, 2004.

A second JPEG compression pass at a different quality factor imprints a
periodic, comb-like structure onto the histogram of a DCT AC
coefficient. This module histograms the DCT(1,1) coefficient across all
8×8 blocks and takes the 1D FFT of that histogram; a strong non-DC
spectral peak indicates the comb pattern characteristic of
double-quantization.

**Fails on:** images that were only ever compressed once (correctly
scores low), and can be confused by unusual single-pass quantization
tables from uncommon encoders.

## 6. Texture Regularity (LBP Entropy)
**File:** `modules/texture_forensics.py::texture_regularity_analysis`

**Citation:** Ojala, Pietikäinen & Mäenpää, *"Multiresolution
Gray-Scale and Rotation Invariant Texture Classification with Local
Binary Patterns,"* IEEE TPAMI, 2002 (LBP itself); applied here to
forensic texture-authenticity checking rather than its original
classification use case.

An 8-neighbour, radius-1 Local Binary Pattern code is computed for
every pixel (implemented with numpy array shifts, not an external CV
library). Real camera micro-texture produces a rich, high-entropy
spread across the 256 possible LBP codes; over-smoothed or up-sampled
synthetic regions often collapse that spread into a narrower band.
Score is based on how far the image's LBP-code Shannon entropy falls
outside an empirically typical natural-photo band (0.62–0.80 of maximum
possible entropy).

**Fails on:** genuinely flat/plain natural regions (clear sky, studio
seamless backdrops) and, at the other extreme, extremely high-ISO noisy
photos — both can sit outside the reference band for legitimate reasons.

## 7. Chromatic Aberration Consistency
**File:** `modules/chromatic_aberration.py::chromatic_aberration_analysis`

Real lenses bend different wavelengths of light by slightly different
amounts, producing a small, physically consistent *radial* misalignment
between colour channels that grows from the image centre toward the
edges. This module samples a grid of patches, finds the best-fit R→G
and B→G integer-pixel shift per patch via normalized cross-correlation
search, and computes the Pearson correlation between shift magnitude and
distance from image centre. Purely synthetic/rendered imagery usually
shows negligible or spatially inconsistent misalignment instead of this
radial pattern.

**Fails on:** images shot on well-corrected apochromatic lenses (very
low aberration to begin with), and on images that have been cropped
tightly around the original centre, removing the radial signal.

## 8. CFA / Demosaicing Footprint
**File:** `modules/cfa_forensics.py::cfa_analysis`

**Citation:** Popescu & Farid, *"Exposing Digital Forgeries in Color
Filter Array Interpolated Images,"* IEEE Trans. Signal Processing, 2005
(simplified proxy of their EM-based periodicity approach).

Real camera sensors sample colour through a periodic (almost always
Bayer) mosaic and reconstruct full resolution via demosaicing
interpolation. That leaves a detectable periodic correlation footprint
at half the sampling frequency (the Nyquist rate of the 2×2 CFA period)
in a simple neighbour-average prediction residual of the green channel.
This module takes the 2D FFT of that residual and checks for a sharp
peak at the Nyquist bins. Purely synthetic imagery generally lacks this
specific footprint.

**This is an explicitly simplified proxy**, not the full EM-based
periodicity estimator from the original paper — documented here rather
than overstated.

**Fails on:** images resized/resampled after the original demosaicing
step (the footprint shifts to a different, harder-to-detect frequency),
and cameras using non-Bayer sensor designs (e.g. Fuji X-Trans).

## 9. Copy-Move Forgery Detection
**File:** `modules/copy_move.py::copy_move_analysis`

**Citation:** Fridrich, Soukal & Lukas, *"Detection of Copy-Move
Forgery in Digital Images,"* Digital Forensic Research Workshop, 2003.

Overlapping blocks are represented by their low-frequency 8×8 DCT
coefficients (a compact, illumination-robust feature), lexicographically
sorted so near-duplicate blocks land next to each other, then compared
within a small neighbourhood in sorted order (avoiding an O(n²) full
comparison). Matching block pairs vote in a shift-vector histogram — a
cloned region produces *many* block pairs sharing the *same* spatial
shift, which is the actual forgery signature (a single matching pair
could just be a coincidence; dozens sharing one consistent shift vector
is not). The dominant shift's matches are drawn on an overlay image.

**Fails on:** cloned regions that were subsequently rotated, scaled, or
significantly recoloured (this implementation matches translation only,
not the full affine-invariant feature matching of more elaborate
SIFT/ORB-based variants), and very low-texture cloned regions (filtered
out deliberately, since flat regions match everywhere and would
otherwise flood the vote histogram with false shift vectors).

## 10. Metadata / EXIF Risk
**File:** `modules/metadata.py::extract_metadata`

Inspects EXIF for camera make/model, capture timestamp, GPS presence,
and software tags matching known editing or generative-AI tool
signatures. Absence of camera EXIF entirely is flagged (common in
AI-generated images and stripped/re-saved files) but explicitly
*not* treated as proof by itself — metadata is trivially strippable,
which is exactly why it carries the lowest fusion weight (8%) of all
ten signals rather than being a gatekeeping check.

---

## Fusion & Verdict

All available signals are combined as a weighted average (weights in
`modules/fusion.py::DEFAULT_WEIGHTS`, chosen to reflect each signal's
relative reliability rather than derived from a labeled validation set —
this is disclosed, not hidden). If a signal reports `insufficient_data`
(most often on very small images), its weight is redistributed
proportionally across the remaining signals rather than defaulting to
zero, so a single inapplicable signal can't silently suppress the score.

```
final_score = Σ(score_i × weight_i) / Σ(weight_i)   for available i
```

Verdict thresholds: **< 35** Likely Authentic · **35–65** Inconclusive
· **≥ 65** Likely Manipulated or AI-Generated.

## Calibration layer (training/)

The fusion weights above are hand-picked, and the earlier version of
this project shipped with *only* that heuristic layer — meaning every
threshold in this system had been tuned against synthetic test images I
generated myself, never validated against a real photo or a real
AI-generated image. That's a real limitation, not a minor caveat, and
it's why `training/` exists.

`training/train_classifier.py` extracts the same ten signals as a
25-dimensional feature vector (`modules/feature_extraction.py`) across a
real labeled dataset and fits a logistic regression / gradient boosting
classifier, reporting held-out test-set ROC-AUC, accuracy, precision,
recall, and a confusion matrix — real evaluation instead of guessed
weights. The logistic regression coefficients are directly interpretable
(`trained/feature_importance.png`): you can point to exactly which of
the ten original signals the calibration step found most/least useful.

`training/train_cnn.py` trains a compact convolutional network (~1.9M
params) **from random initialization** directly on image pixels,
following the augmentation strategy from Wang, Wang, Zhang, Owens &
Efros, *"CNN-generated images are surprisingly easy to spot... for
now,"* CVPR 2020 (random JPEG re-compression + Gaussian blur during
training, to prevent the network from learning a compression-artifact
shortcut instead of genuine generative fingerprints). This typically
has a higher accuracy ceiling than the feature-based classifier
specifically for whole-image AI-generation detection, at the cost of
interpretability.

Both are still fully original: the weights that come out belong to
whoever trains them, on data they chose — nothing is downloaded from a
model hub. The app detects a `trained/` and/or `trained_cnn/` folder
automatically and shows a clearly labeled **calibrated verdict**
alongside the original heuristic breakdown; with neither present, the
app runs exactly as the heuristic-only version did, with an explicit
on-screen note that the verdict is uncalibrated.


## Why fusion, not one "best" signal

Every technique above has a documented failure mode above — that's
inherent to the field, not a flaw in this implementation. A single
signal claiming 99% accuracy is either overfit to its own test
distribution or hasn't met an adversarial case yet. Combining ten
independent, differently-motivated signals — statistical (Benford),
physical (chromatic aberration, CFA), geometric (copy-move), and
compression-domain (ELA, double-compression) — means an attacker has to
defeat all ten simultaneously, not fool one classifier. This is the
same principle real forensic laboratories operate on, and it is the
central design decision of this system.
