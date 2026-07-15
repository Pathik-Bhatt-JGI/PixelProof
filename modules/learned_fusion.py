"""Loads a classifier trained by training/train_classifier.py (if
present) and produces a calibrated probability from the same 25
hand-engineered features the rest of the app already computes.

This is intentionally optional and gracefully absent: a fresh checkout
of this repo has no trained/ folder, so the app runs exactly as before
(heuristic 10-signal fusion only) until the user trains a model on real
data with training/train_classifier.py. Once trained/ exists, the app
additionally shows a "Calibrated ML Verdict" derived from real labeled
data — distinct from, and clearly labeled apart from, the explainable
heuristic breakdown.
"""
import json
from pathlib import Path

TRAINED_DIR = Path(__file__).parent.parent / "trained"


def is_available() -> bool:
    return (TRAINED_DIR / "metrics.json").exists() and (TRAINED_DIR / "scaler.joblib").exists()


def load_model():
    import joblib
    with open(TRAINED_DIR / "metrics.json") as f:
        meta = json.load(f)
    model_name = meta["recommended_model"]
    model = joblib.load(TRAINED_DIR / f"{model_name}.joblib")
    scaler = joblib.load(TRAINED_DIR / "scaler.joblib")
    with open(TRAINED_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    return {
        "model": model, "scaler": scaler, "feature_names": feature_names,
        "model_name": model_name, "metrics": meta["results"][model_name],
        "n_train": meta.get("n_train", 0), "n_val": meta.get("n_val", 0), "n_test": meta.get("n_test", 0),
    }


def predict(feats: dict, loaded: dict) -> dict:
    vector = [[feats[name] for name in loaded["feature_names"]]]
    scaled = loaded["scaler"].transform(vector)
    prob_fake = float(loaded["model"].predict_proba(scaled)[0, 1])
    return {
        "score": prob_fake * 100,
        "model_name": loaded["model_name"],
        "test_roc_auc": loaded["metrics"]["test_roc_auc"],
        "test_accuracy": loaded["metrics"]["test_accuracy"],
        "n_test": loaded["n_test"],
        "n_train": loaded["n_train"],
        "low_confidence": loaded["n_test"] < 100,
    }
