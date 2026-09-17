"""Part 3.1 of 4 - CNN classifier for the LFW faces (PyTorch).

Architecture (as requested by the lab sheet: two 3x3 convolution layers with 32 filters each,
followed by dense layers):

    Conv2d(1 -> 32, 3x3, pad 1) - BatchNorm - ReLU - MaxPool(2)     50x37 -> 25x18
    Conv2d(32 -> 32, 3x3, pad 1) - BatchNorm - ReLU - MaxPool(2)    25x18 -> 12x9
    Flatten - Linear(32*12*9 -> 128) - ReLU - Dropout(0.5) - Linear(128 -> n_classes)

Trained with Adam and (sparse) categorical cross-entropy. The same 75/25 split (random_state=42)
as Part 2 is used so the accuracy is directly comparable with the PCA + Random Forest baseline.

Run from the repository root:
    python part3_cnn/lfw_cnn.py --epochs 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `common` importable

from sklearn.datasets import fetch_lfw_people  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from common import RESULTS_DIR, ensure_dir, get_device, set_seed  # noqa: E402
from common.device import describe_device  # noqa: E402
from common.plotting import plot_curves  # noqa: E402

OUT_DIR = RESULTS_DIR / "part3_cnn"


class LFWConvNet(nn.Module):
    """Two 3x3/32-filter convolution blocks followed by a small dense classifier."""

    def __init__(self, in_shape: tuple[int, int, int], n_classes: int, n_filters: int = 32, hidden: int = 128, dropout: float = 0.5):
        super().__init__()
        c, h, w = in_shape
        self.features = nn.Sequential(
            nn.Conv2d(c, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():  # infer the flattened size instead of hard-coding 32*12*9
            n_flat = self.features(torch.zeros(1, c, h, w)).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def load_data(device: torch.device, batch_size: int):
    """LFW images as 4-D tensors [N, 1, H, W] in [0, 1]; same split as Part 2."""
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    X, Y = lfw_people.images, lfw_people.target
    print(f"X_min: {X.min():.3f}  X_max: {X.max():.3f}  (already scaled to [0, 1]; no normalisation needed)")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    X_train = X_train[:, np.newaxis, :, :]  # conv layers expect [N, C, H, W]
    X_test = X_test[:, np.newaxis, :, :]
    print(f"X_train shape: {X_train.shape}   X_test shape: {X_test.shape}")

    to_t = lambda a, dt: torch.as_tensor(a, dtype=dt)  # noqa: E731
    train_loader = DataLoader(TensorDataset(to_t(X_train, torch.float32), to_t(y_train, torch.long)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(to_t(X_test, torch.float32), to_t(y_test, torch.long)), batch_size=256)
    return train_loader, test_loader, X_train.shape[1:], lfw_people.target_names, y_test


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, np.ndarray]:
    """Return (mean loss, accuracy, predictions) over ``loader``."""
    model.eval()
    total_loss, correct, preds = 0.0, 0, []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        preds.append(pred.cpu())
    n = len(loader.dataset)
    return total_loss / n, correct / n, torch.cat(preds).numpy()


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss, correct = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
        correct += (logits.argmax(1) == yb).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    set_seed(args.seed)
    ensure_dir(OUT_DIR)
    device = get_device(args.device)
    print(f"device: {describe_device(device)}")

    train_loader, test_loader, in_shape, target_names, y_test = load_data(device, args.batch_size)
    model = LFWConvNet(in_shape, n_classes=len(target_names)).to(device)
    print(model)
    print(f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: dict[str, list[float]] = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc, _ = evaluate(model, test_loader, device)
        for k, v in zip(history, (train_loss, test_loss, train_acc, test_acc)):
            history[k].append(v)
        print(f"epoch {epoch:3d}/{args.epochs}  train loss {train_loss:.4f} acc {train_acc:.3f} | test loss {test_loss:.4f} acc {test_acc:.3f}")
    print(f"training time: {time.time() - start:.1f}s")

    _, test_acc, predictions = evaluate(model, test_loader, device)
    report = classification_report(y_test, predictions, target_names=target_names)
    print(f"\nFinal test accuracy: {test_acc:.4f}\n{report}")

    plot_curves({"train": history["train_loss"], "test": history["test_loss"]}, OUT_DIR / "lfw_cnn_loss.png", "epoch", "cross-entropy", "LFW CNN loss")
    plot_curves({"train": history["train_acc"], "test": history["test_acc"]}, OUT_DIR / "lfw_cnn_accuracy.png", "epoch", "accuracy", "LFW CNN accuracy")
    torch.save(model.state_dict(), OUT_DIR / "lfw_cnn.pt")
    with open(OUT_DIR / "lfw_cnn_results.json", "w") as f:
        json.dump({"test_accuracy": float(test_acc), "history": history, "report": report, "args": vars(args)}, f, indent=2)
    print(f"[saved] {OUT_DIR / 'lfw_cnn_results.json'}")


if __name__ == "__main__":
    main()
