"""
Train a calibrated classifier on top of ForensiQ's ten original forensic
signals, using real labeled data. Fixes the core problem with the
heuristic-only version: every threshold in modules/*.py was hand-picked
against synthetic test images, not validated against real photos or
real AI-generated images. This script replaces guesswork with a model
fit and evaluated on actual ground truth.

USAGE
-----
1. Download a labeled real-vs-AI dataset (see ../DATASET_GUIDE.md for
   specific recommendations). Organize it as:

       your_dataset/
           real/   *.jpg | *.png | ...
           fake/   *.jpg | *.png | ...

2. Run:
       python training/train_classifier.py --data-dir your_dataset --output-dir trained

3. Drop the resulting `trained/` folder into the ForensiQ project root.
   The app will automatically detect it and show a calibrated ML
   verdict alongside the original explainable signal breakdown.

This trains on the same 25 hand-engineered features produced by
modules/feature_extraction.py — i.e. it calibrates the *combination* of
our original signals against real data rather than replacing them with
a black box. The logistic regression coefficients are directly
interpretable ("how much did each of my forensic signals matter").
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.feature_extraction import FEATURE_NAMES, extract_feature_vector, feature_dict_to_vector


def collect_files(data_dir: Path, max_per_class: int | None):
    real_dir, fake_dir = data_dir / "real", data_dir / "fake"
    if not real_dir.exists() or not fake_dir.exists():
        raise SystemExit(
            f"Expected '{data_dir}/real/' and '{data_dir}/fake/' subfolders. "
            f"Found: real={real_dir.exists()}, fake={fake_dir.exists()}"
        )
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
    real_files = sorted(p for p in real_dir.rglob("*") if p.suffix.lower() in exts)
    fake_files = sorted(p for p in fake_dir.rglob("*") if p.suffix.lower() in exts)
    if max_per_class:
        real_files = real_files[:max_per_class]
        fake_files = fake_files[:max_per_class]
    return real_files, fake_files


def extract_dataset_features(files: list, label: int):
    from PIL import Image
    from tqdm import tqdm

    X, y, failed = [], [], 0
    for path in tqdm(files, desc=f"label={label}"):
        try:
            img = Image.open(path)
            img.load()
            feats = extract_feature_vector(img)
            X.append(feature_dict_to_vector(feats))
            y.append(label)
        except Exception as e:
            failed += 1
            continue
    if failed:
        print(f"  ({failed} files failed to process and were skipped)")
    return X, y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path, help="Folder containing real/ and fake/ subfolders")
    ap.add_argument("--output-dir", default=Path("trained"), type=Path)
    ap.add_argument("--max-per-class", type=int, default=None, help="Cap images per class (useful for a quick test run)")
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                                  recall_score, f1_score, confusion_matrix, roc_curve)
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real_files, fake_files = collect_files(args.data_dir, args.max_per_class)
    print(f"Found {len(real_files)} real images, {len(fake_files)} fake images.")
    if len(real_files) < 50 or len(fake_files) < 50:
        print("WARNING: fewer than 50 images in one class — results will not be statistically meaningful. "
              "This is fine for a smoke-test run, not for a real evaluation.")

    t0 = time.time()
    X_real, y_real = extract_dataset_features(real_files, label=0)
    X_fake, y_fake = extract_dataset_features(fake_files, label=1)
    print(f"Feature extraction took {time.time()-t0:.1f}s for {len(X_real)+len(X_fake)} images.")

    X = np.array(X_real + X_fake)
    y = np.array(y_real + y_fake)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=args.test_size + args.val_size, stratify=y, random_state=args.seed)
    rel_test = args.test_size / (args.test_size + args.val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test, stratify=y_temp, random_state=args.seed)
    print(f"Split: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s, X_test_s = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

    models = {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=args.seed),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    best_name, best_auc = None, -1

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        val_probs = model.predict_proba(X_val_s)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)
        print(f"[{name}] validation ROC-AUC: {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc, best_name = val_auc, name

        test_probs = model.predict_proba(X_test_s)[:, 1]
        test_preds = (test_probs >= 0.5).astype(int)
        results[name] = {
            "val_roc_auc": float(val_auc),
            "test_roc_auc": float(roc_auc_score(y_test, test_probs)),
            "test_accuracy": float(accuracy_score(y_test, test_preds)),
            "test_precision": float(precision_score(y_test, test_preds)),
            "test_recall": float(recall_score(y_test, test_preds)),
            "test_f1": float(f1_score(y_test, test_preds)),
            "confusion_matrix": confusion_matrix(y_test, test_preds).tolist(),
        }
        joblib.dump(model, args.output_dir / f"{name}.joblib")

    joblib.dump(scaler, args.output_dir / "scaler.joblib")
    with open(args.output_dir / "feature_names.json", "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump({"results": results, "recommended_model": best_name,
                    "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test)}, f, indent=2)

    # --- ROC curve plot ---
    plt.figure(figsize=(5.5, 5), facecolor="#060C12")
    ax = plt.gca()
    ax.set_facecolor("#060C12")
    for name, model in models.items():
        probs = model.predict_proba(X_test_s)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC={results[name]['test_roc_auc']:.3f})", linewidth=1.6)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#3A5570", linewidth=1)
    ax.set_xlabel("False Positive Rate", color="#7A95B0")
    ax.set_ylabel("True Positive Rate", color="#7A95B0")
    ax.set_title("ROC Curve (held-out test set)", color="#E8F0F8")
    ax.tick_params(colors="#7A95B0")
    for spine in ax.spines.values():
        spine.set_color("#0F2030")
    ax.legend(facecolor="#060C12", edgecolor="#0F2030", labelcolor="#7A95B0", fontsize=8)
    ax.grid(True, color="#0F2030", linewidth=0.6)
    plt.tight_layout()
    plt.savefig(args.output_dir / "roc_curve.png", dpi=140, facecolor="#060C12")
    plt.close()

    # --- feature importance (logistic regression coefficients) ---
    lr = models["logistic_regression"]
    coefs = lr.coef_[0]
    order = np.argsort(np.abs(coefs))[::-1]
    plt.figure(figsize=(7, 6), facecolor="#060C12")
    ax = plt.gca()
    ax.set_facecolor("#060C12")
    names_sorted = [FEATURE_NAMES[i] for i in order]
    vals_sorted = [coefs[i] for i in order]
    colors = ["#FF3366" if v > 0 else "#00FFB3" for v in vals_sorted]
    ax.barh(range(len(names_sorted)), vals_sorted, color=colors)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=7, color="#7A95B0")
    ax.invert_yaxis()
    ax.set_xlabel("Logistic regression coefficient (red = pushes toward FAKE)", color="#7A95B0")
    ax.set_title("Learned Feature Importance", color="#E8F0F8")
    ax.tick_params(colors="#7A95B0")
    for spine in ax.spines.values():
        spine.set_color("#0F2030")
    ax.grid(True, axis="x", color="#0F2030", linewidth=0.6)
    plt.tight_layout()
    plt.savefig(args.output_dir / "feature_importance.png", dpi=140, facecolor="#060C12")
    plt.close()

    print("\n" + "=" * 60)
    print(f"Recommended model: {best_name} (validation ROC-AUC={best_auc:.4f})")
    print(f"Test-set metrics: {json.dumps(results[best_name], indent=2)}")
    print(f"Saved to: {args.output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
