"""Optional inference wrapper for a CNN trained by training/train_cnn.py.

Kept fully optional and gracefully absent: torch is NOT a dependency of
the base app (requirements.txt stays lightweight). If the user has
trained a CNN and installed torch, and a trained_cnn/model.pt checkpoint
exists, the app will additionally show a CNN-based signal. Otherwise
this module quietly reports itself unavailable and the rest of the app
is unaffected.
"""
import json
from pathlib import Path

TRAINED_CNN_DIR = Path(__file__).parent.parent / "trained_cnn"


def is_available() -> bool:
    if not (TRAINED_CNN_DIR / "model.pt").exists():
        return False
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError:
        return False
    return True


def load_model():
    import torch
    from .cnn_arch import build_model
    checkpoint = torch.load(TRAINED_CNN_DIR / "model.pt", map_location="cpu")
    model = build_model()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    metrics, n_train, n_val, n_test = {}, 0, 0, 0
    metrics_path = TRAINED_CNN_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        n_train = metrics.get("n_train", 0)
        n_val = metrics.get("n_val", 0)
        n_test = metrics.get("n_test", 0)

    return {
        "model": model, "img_size": checkpoint.get("img_size", 160),
        "metrics": metrics, "n_train": n_train, "n_val": n_val, "n_test": n_test,
    }


def predict(image, loaded: dict) -> dict:
    import torch
    from torchvision import transforms

    tfm = transforms.Compose([
        transforms.Resize((loaded["img_size"], loaded["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    x = tfm(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        prob_fake = float(torch.softmax(loaded["model"](x), dim=1)[0, 1])

    return {
        "score": prob_fake * 100,
        "test_roc_auc": loaded["metrics"].get("test_roc_auc"),
        "test_accuracy": loaded["metrics"].get("test_accuracy"),
        "n_test": loaded["n_test"],
        "n_train": loaded["n_train"],
        "low_confidence": loaded["n_test"] < 100,
    }
