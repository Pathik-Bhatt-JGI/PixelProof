"""
Train a compact convolutional network FROM SCRATCH (random init — no
downloaded/pretrained weights of any kind) to detect AI-generated
images directly from pixels. This is the higher-accuracy-ceiling
counterpart to train_classifier.py: the hand-engineered forensic
signals in modules/*.py were built for classical splice/tamper
detection, and while they're genuinely useful and fully explainable,
a CNN trained directly on real vs. AI-generated pixel data will
typically outperform them specifically at the "is this whole image
synthetic" task, because it can learn generator-specific statistical
fingerprints no hand-designed filter anticipated.

This is still entirely "your own model": you choose the data, you run
the training, the weights that come out belong to you. Nothing is
downloaded from Hugging Face or any model hub.

Includes the augmentation strategy from Wang, Wang, Zhang, Owens & Efros,
"CNN-generated images are surprisingly easy to spot... for now," CVPR
2020 — random JPEG re-compression and Gaussian blur during training —
which prevents the network from learning a trivial compression-artifact
shortcut instead of genuine generative fingerprints, and measurably
improves generalization to generators not seen during training.

USAGE
-----
    pip install torch torchvision
    python training/train_cnn.py --data-dir your_dataset --epochs 15

Expects the same folder layout as train_classifier.py:
    your_dataset/real/*.jpg
    your_dataset/fake/*.jpg
"""
import argparse
import io
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


# ------------------------------------------------------------------------
# The model architecture lives in modules/cnn_arch.py and is imported
# from there (not redefined here) so that training and inference in
# modules/cnn_detector.py can never silently drift apart.
# ------------------------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent))
from modules.cnn_arch import build_model  # noqa: E402


# --------------------------------------------------------- augmentation --
class RandomJPEGCompression:
    """Re-encodes through JPEG at a random quality — forces the network
    to look past compression artifacts rather than keying on them."""

    def __init__(self, quality_range=(65, 100), p=0.5):
        self.quality_range, self.p = quality_range, p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        q = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomGaussianBlur:
    def __init__(self, radius_range=(0, 1.2), p=0.3):
        self.radius_range, self.p = radius_range, p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        r = random.uniform(*self.radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius=r))


def build_transforms(img_size: int, train: bool):
    from torchvision import transforms
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            RandomJPEGCompression(),
            RandomGaussianBlur(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--output-dir", default=Path("trained_cnn"), type=Path)
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--test-split", type=float, default=0.15)
    ap.add_argument("--max-per-class", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=4, help="Early-stopping patience (epochs without val AUC improvement)")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import ImageFolder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    real_dir, fake_dir = args.data_dir / "real", args.data_dir / "fake"
    if not real_dir.exists() or not fake_dir.exists():
        raise SystemExit(f"Expected '{args.data_dir}/real/' and '{args.data_dir}/fake/' subfolders.")

    # torchvision ImageFolder wants class subfolders directly under data_dir;
    # 'real' sorts before 'fake' alphabetically -> class 0 = fake, class 1 = real.
    # We relabel explicitly below so this is robust regardless of folder order.
    full_train_ds = ImageFolder(args.data_dir, transform=build_transforms(args.img_size, train=True))
    full_eval_ds = ImageFolder(args.data_dir, transform=build_transforms(args.img_size, train=False))
    class_to_idx = full_train_ds.class_to_idx
    fake_class_idx = class_to_idx.get("fake")
    print(f"Classes: {class_to_idx} (label 1 = fake/AI-generated)")

    targets = np.array(full_train_ds.targets)
    if args.max_per_class:
        keep_idx = []
        for c in np.unique(targets):
            c_idx = np.where(targets == c)[0][:args.max_per_class]
            keep_idx.extend(c_idx.tolist())
        keep_idx = np.array(keep_idx)
    else:
        keep_idx = np.arange(len(targets))

    labels = (targets[keep_idx] == fake_class_idx).astype(int)
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        keep_idx, labels, test_size=args.val_split + args.test_split, stratify=labels, random_state=args.seed)
    rel_test = args.test_split / (args.val_split + args.test_split)
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=rel_test, stratify=y_temp, random_state=args.seed)
    print(f"Split: {len(idx_train)} train / {len(idx_val)} val / {len(idx_test)} test")

    train_ds = Subset(full_train_ds, idx_train)
    val_ds = Subset(full_eval_ds, idx_val)
    test_ds = Subset(full_eval_ds, idx_test)

    def relabel_collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.tensor([1 if b[1] == fake_class_idx else 0 for b in batch], dtype=torch.long)
        return xs, ys

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=relabel_collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=relabel_collate, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=relabel_collate, num_workers=0)

    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_val_loss, best_state, patience_ctr = float("inf"), None, 0

    for epoch in range(args.epochs):
        model.train()
        t0, running_loss = time.time(), 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_ds)
        scheduler.step()

        model.eval()
        val_loss, val_probs, val_labels = 0.0, [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += criterion(out, yb).item() * xb.size(0)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs.tolist())
                val_labels.extend(yb.cpu().numpy().tolist())
        val_loss /= len(val_ds)
        val_auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else 0.5

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        print(f"Epoch {epoch+1}/{args.epochs} — train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_auc={val_auc:.4f} ({time.time()-t0:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss, best_state, patience_ctr = val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"Early stopping (no val_loss improvement for {args.patience} epochs).")
                break

    model.load_state_dict(best_state)

    model.eval()
    test_probs, test_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            probs = torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy()
            test_probs.extend(probs.tolist())
            test_labels.extend(yb.numpy().tolist())
    test_preds = [1 if p >= 0.5 else 0 for p in test_probs]

    metrics = {
        "test_roc_auc": float(roc_auc_score(test_labels, test_probs)),
        "test_accuracy": float(accuracy_score(test_labels, test_preds)),
        "test_precision": float(precision_score(test_labels, test_preds)),
        "test_recall": float(recall_score(test_labels, test_preds)),
        "test_f1": float(f1_score(test_labels, test_preds)),
        "confusion_matrix": confusion_matrix(test_labels, test_preds).tolist(),
        "best_val_loss": float(best_val_loss),
        "n_params": n_params,
        "img_size": args.img_size,
        "epochs_trained": len(history["train_loss"]),
        "n_train": len(idx_train),
        "n_val": len(idx_val),
        "n_test": len(idx_test),
    }

    torch.save({"state_dict": model.state_dict(), "img_size": args.img_size}, args.output_dir / "model.pt")
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(args.output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # training curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#060C12")
    for ax in axes:
        ax.set_facecolor("#060C12")
        ax.tick_params(colors="#7A95B0")
        for spine in ax.spines.values():
            spine.set_color("#0F2030")
        ax.grid(True, color="#0F2030", linewidth=0.6)
    axes[0].plot(history["train_loss"], label="train", color="#00FFB3")
    axes[0].plot(history["val_loss"], label="val", color="#FF3366")
    axes[0].set_title("Loss", color="#E8F0F8")
    axes[0].legend(facecolor="#060C12", edgecolor="#0F2030", labelcolor="#7A95B0")
    axes[1].plot(history["val_auc"], color="#00B8FF")
    axes[1].set_title("Validation ROC-AUC", color="#E8F0F8")
    plt.tight_layout()
    plt.savefig(args.output_dir / "training_curves.png", dpi=140, facecolor="#060C12")
    plt.close()

    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    plt.figure(figsize=(5.5, 5), facecolor="#060C12")
    ax = plt.gca()
    ax.set_facecolor("#060C12")
    ax.plot(fpr, tpr, color="#00FFB3", linewidth=1.8, label=f"CNN (AUC={metrics['test_roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#3A5570", linewidth=1)
    ax.set_xlabel("False Positive Rate", color="#7A95B0")
    ax.set_ylabel("True Positive Rate", color="#7A95B0")
    ax.set_title("CNN ROC Curve (held-out test set)", color="#E8F0F8")
    ax.tick_params(colors="#7A95B0")
    for spine in ax.spines.values():
        spine.set_color("#0F2030")
    ax.legend(facecolor="#060C12", edgecolor="#0F2030", labelcolor="#7A95B0")
    ax.grid(True, color="#0F2030", linewidth=0.6)
    plt.tight_layout()
    plt.savefig(args.output_dir / "roc_curve.png", dpi=140, facecolor="#060C12")
    plt.close()

    print("\n" + "=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Test-set metrics: {json.dumps(metrics, indent=2)}")
    print(f"Saved to: {args.output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
