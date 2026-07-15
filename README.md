# ForensiQ — Original Image Authentication Engine

An explainable, multi-signal image forensics tool for detecting AI-generated
and manipulated images. **Every detector is self-implemented signal
processing — no pretrained AI models, no third-party classifiers, no
external weights of any kind.** Everything runs offline, on CPU, in under
a second per image.

This is built to be defended in an interview or a viva: every signal has
a named academic citation, a documented failure mode, and a unit test.
See **`METHODOLOGY.md`** for the full technical writeup with citations —
read that before a technical interview about this project.

## Calibrating against real data (do this before an interview)

**The heuristic scores above are hand-tuned, not validated.** I built
and tested every signal against synthetic images I generated myself —
that's enough to confirm each algorithm's *mechanism* works (e.g. the
copy-move detector finds an exact planted shift vector), but it is
**not** the same as validated accuracy on real photos and real
AI-generated images. Don't claim otherwise.

`training/` fixes this with two scripts you run locally against a real
labeled dataset:

```bash
pip install -r training/requirements.txt
# see training/DATASET_GUIDE.md for where to get a real dataset

python training/train_classifier.py --data-dir your_dataset   # fast, interpretable
python training/train_cnn.py --data-dir your_dataset --epochs 15   # slower, higher accuracy ceiling
```

Both report real held-out test-set ROC-AUC, accuracy, precision,
recall, and a confusion matrix — copy those numbers into your report
verbatim. Drop the resulting `trained/` and/or `trained_cnn/` folder
into the project root and the app automatically shows a **Calibrated
Verdict** (validated on real data) alongside the original explainable
10-signal breakdown. With neither folder present, the app runs exactly
as before, with an on-screen note that the verdict is uncalibrated —
it never silently pretends to be more validated than it is.

See `training/DATASET_GUIDE.md` for specific free dataset recommendations
and `METHODOLOGY.md` for the full writeup of what each script does and why.

## What it does

Ten independent, from-scratch forensic signals, fused into one weighted
confidence score:

| # | Signal | Technique | Citation |
|---|---|---|---|
| 1 | Error Level Analysis | JPEG re-compression differencing | — |
| 2 | Frequency-Domain (FFT) | Radial spectral energy analysis | — |
| 3 | PRNU Sensor Noise | Wavelet-domain Wiener filtering | Lukas, Fridrich & Goljan, 2006 |
| 4 | Benford's Law (DCT) | Leading-digit distribution of DCT coefficients | Fu, Shi & Su, 2007 |
| 5 | Double-Compression | DCT histogram spectral periodicity | Popescu & Farid, 2004 |
| 6 | Texture Regularity | From-scratch Local Binary Pattern entropy | Ojala et al., 2002 |
| 7 | Chromatic Aberration | Patch-wise cross-correlation vs. radial position | — |
| 8 | CFA / Demosaicing Footprint | Nyquist-frequency periodicity in green-channel residual | Popescu & Farid, 2005 |
| 9 | Copy-Move Forgery | Block-based DCT feature matching + shift-vector voting | Fridrich, Soukal & Lukas, 2003 |
| 10 | Metadata / EXIF Risk | File inspection | — |

Plus: **SHA-256/MD5 evidence hashing**, a **chain-of-custody log**, a
**composite manipulation-localization heatmap** (blends ELA + PRNU into
one overlay), per-signal visual explainability for every detector, and a
**one-click PDF forensic report**.

### Why fusion instead of one signal?
No single forensic technique is reliable on its own — every technique in
`METHODOLOGY.md` has a documented failure mode. Combining ten
independent, differently-motivated signals (statistical, physical,
geometric, compression-domain) means a forgery has to defeat all ten
simultaneously. That's the same principle real forensic labs operate on,
and it's a stronger, more defensible claim in a viva or a product pitch
than "99% accuracy from one model."

## Quick start (local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). No
internet connection is required at any point.

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

20 unit tests validate hash determinism, fusion weight-redistribution
arithmetic, LBP uniform-pattern classification against the textbook
definition, Benford digit extraction, and — the two that matter most —
that the copy-move detector finds the *exact* known shift vector in a
synthetically forged test image and that the PRNU module's inconsistency
score rises on a synthetic noise-floor splice.

## Deploying online for free

### Option A — Streamlit Community Cloud (recommended, easiest)
1. Push this folder to a new GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click "New app," pick the repo and `app.py`.
3. Deploy. You get a free public URL like `yourapp.streamlit.app`.
4. No GPU, no large model downloads — comfortably fits the free tier.

### Option B — Hugging Face Spaces
1. Create a new [Space](https://huggingface.co/new-space) → SDK: **Streamlit**.
2. Push this folder's contents to the Space's git repo.
3. Free CPU tier is more than sufficient.

## Project structure

```
forensiq-lite/
├── app.py                       # Streamlit UI + orchestration (custom themed)
├── requirements.txt
├── METHODOLOGY.md                # Full technical writeup with citations
├── assets/
│   └── style.css                  # Visual theme
├── modules/
│   ├── hashing.py                  # SHA-256/MD5 evidence hashing
│   ├── metadata.py                 # EXIF extraction + AI-tool signature flags
│   ├── forensics.py                # ELA, FFT frequency analysis
│   ├── prnu.py                     # Wavelet-domain PRNU sensor noise (Lukas-Fridrich-Goljan)
│   ├── dct_forensics.py            # Benford's Law + double-compression
│   ├── texture_forensics.py        # From-scratch LBP texture regularity
│   ├── chromatic_aberration.py     # From-scratch lens-physics consistency check
│   ├── cfa_forensics.py            # CFA/demosaicing footprint detection
│   ├── copy_move.py                # Block-matching copy-move forgery detection
│   ├── localization.py             # Composite manipulation heatmap overlay
│   ├── feature_extraction.py       # Bridges the 10 signals -> ML feature vector
│   ├── learned_fusion.py           # Loads a trained/ classifier if present (optional)
│   ├── cnn_arch.py                 # Shared CNN architecture (train + inference)
│   ├── cnn_detector.py             # Loads a trained_cnn/ model if present (optional)
│   ├── charts.py                   # Dark-themed matplotlib chart rendering
│   ├── fusion.py                   # Weighted ensemble scoring + verdict thresholds
│   ├── ui.py                       # Themed HTML component helpers
│   └── report.py                   # PDF forensic report generation (ReportLab)
├── training/
│   ├── train_classifier.py         # Calibrate fusion weights on real labeled data
│   ├── train_cnn.py                # Train a from-scratch CNN on real labeled data
│   ├── DATASET_GUIDE.md            # Where to get real data
│   └── requirements.txt            # scikit-learn, torch, etc. (training only)
├── tests/
│   └── test_modules.py             # 20 unit tests (pytest)
```

## Honest limitations (say this in your report/viva/pitch)

- **The heuristic layer alone is uncalibrated** — its thresholds came
  from synthetic test images, not real validated data. Say plainly
  whether you've run the `training/` calibration step for your specific
  submission, and if so, quote its real test-set numbers (accuracy,
  ROC-AUC, confusion matrix) rather than any number from this README.
- **No forensic technique is 100% reliable on its own** — that's exactly
  why this is a 10-signal fusion system. Each signal's specific failure
  mode is documented in `METHODOLOGY.md`; know them before your interview.
- Metadata analysis is easily defeated by anyone who strips EXIF data —
  it carries the lowest fusion weight (8%) for exactly this reason.
- The copy-move detector matches translation only, not rotation/scaling —
  documented as a known limitation vs. full affine-invariant approaches.
- The CFA analysis is an explicitly simplified proxy of the full EM-based
  method in the original paper — say so if asked, don't overclaim it.
- Heavily downscaled, re-compressed, or screenshotted images will show
  weaker/absent DCT and chromatic-aberration signals through no fault of
  the source; the fusion engine reweights automatically when a signal
  reports insufficient data, and this is disclosed in the UI.
- This is the **image module** (as originally scoped). The same
  from-scratch architecture extends to video (frame-by-frame + temporal
  consistency) and audio (spectral/prosody analysis) if you build further.

## Extending it

- **Video:** extract frames with OpenCV, run this same signal pipeline
  per sampled frame, add a temporal-consistency score across frames.
- **Audio:** build an original spectral/prosody analysis module (pitch
  jitter, formant consistency, silence/breathing statistics) following
  the same self-implemented pattern as the modules here.
- **Copy-move upgrade:** swap the DCT block-feature matching for
  SIFT/ORB keypoint matching plus a RANSAC affine-transform fit, to
  catch rotated/scaled clones — a natural "v2" talking point.
- **Batch/case management:** swap the single-file uploader for a folder
  uploader plus a SQLite case database.
