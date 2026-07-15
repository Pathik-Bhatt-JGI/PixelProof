# Dataset Guide

Both training scripts expect this folder layout:

```
your_dataset/
    real/    *.jpg | *.png | ...
    fake/    *.jpg | *.png | ...
```

`real` = authentic camera/phone photos. `fake` = AI-generated or
manipulated images. Balanced classes (similar counts in each folder)
work best — both scripts will warn you if either class has under 50
images, which is fine for a smoke test but not for a real evaluation.

I can't download these into the sandbox I work in (network access there
is restricted to package registries, not dataset hosts), so grab one of
these yourself — both are free, well-known, and don't require anything
beyond a free account:

## Recommended: "140k Real and Fake Faces" (Kaggle)
- ~140,000 images at 256×256 — real faces from Flickr-Faces-HQ (FFHQ),
  fake faces from thispersondoesnotexist.com (StyleGAN).
- Good resolution for every signal in this pipeline — copy-move,
  chromatic aberration, and CFA analysis all need reasonable image size
  to work well, and 256×256 comfortably clears that bar.
- `kaggle datasets download -d xhlulu/140k-real-and-fake-faces`

## Fast/easy alternative: "CIFAKE" (Kaggle)
- 120,000 images, 60k real (CIFAR-10) vs. 60k AI-generated (Stable
  Diffusion), by Bird & Lotfi.
- Very fast to train on — good for a first smoke-test run.
- **Caveat:** images are only 32×32. Several of this pipeline's signals
  (copy-move block-matching, chromatic-aberration patch sampling, CFA
  analysis) need more pixels than that to produce a reliable reading and
  will report `insufficient_data` on most of these images — the feature
  vector will have a lot of zeros. Fine for a fast first pass on the
  simpler signals (ELA, frequency, texture, Benford), not ideal for the
  full feature set.
- `kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images`

## More modern / advanced: GenImage or ArtiFact
Cover a wider range of 2023+ generators (SDXL, Midjourney, etc.) at
higher resolution. Larger and more involved to set up — worth it if you
want your model to generalize to the newest generators specifically,
but start with one of the two datasets above first to get the pipeline
working end to end.

## After downloading
1. Extract/organize into the `real/` / `fake/` layout above.
2. Quick pipeline check (a minute or two):
   ```bash
   python training/train_classifier.py --data-dir your_dataset --max-per-class 100
   ```
3. Full run once that works cleanly:
   ```bash
   python training/train_classifier.py --data-dir your_dataset
   python training/train_cnn.py --data-dir your_dataset --epochs 15
   ```
4. Copy the resulting `trained/` and/or `trained_cnn/` folders into the
   ForensiQ project root (next to `app.py`). The app detects them
   automatically on next launch and shows a calibrated verdict with
   real test-set accuracy, on top of the existing explainable signal
   breakdown.

## A note on what "good accuracy" looks like
Report your test-set ROC-AUC and accuracy exactly as the scripts print
them — don't round up or cherry-pick. A model that gets ~85-92% test
accuracy on a dataset like this, with an honestly reported confusion
matrix, is a far stronger thing to bring to an interview than a claimed
99% with no evaluation behind it. If a generator wasn't in your training
data, expect real degradation on images from it — say so; it's expected,
not a flaw.
