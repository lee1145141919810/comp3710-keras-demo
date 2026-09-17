"""Part 2 of 4 - Eigenfaces: PCA of the LFW faces (NumPy SVD) + Random Forest classification.

Pipeline
    1. Load the funnelled "Labeled Faces in the Wild" dataset (people with >= 70 images, 0.4 scale).
    2. Stratify-free random split into 75 % train / 25 % test (random_state=42 as in the lab sheet).
    3. Centre the data with the *training* mean and compute the PCA basis with an SVD.
    4. Plot the eigenfaces (principal components) and the compactness (cumulative explained variance).
    5. Project both splits into "face space" and train a Random Forest on the projected features.
    6. Report accuracy and a per-class classification report.

Run from the repository root:
    python part2_eigenfaces/eigenfaces.py --n_components 150
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `common` importable

import matplotlib.pyplot as plt  # noqa: E402
from sklearn.datasets import fetch_lfw_people  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from common import RESULTS_DIR, ensure_dir, save_figure, set_seed  # noqa: E402

OUT_DIR = RESULTS_DIR / "part2_eigenfaces"


def load_lfw(min_faces_per_person: int = 70, resize: float = 0.4):
    """Download (once) and return the LFW people dataset used throughout Parts 2 and 3."""
    lfw_people = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=resize)
    n_samples, h, w = lfw_people.images.shape
    print("Total dataset size:")
    print(f"  n_samples : {n_samples}")
    print(f"  n_features: {lfw_people.data.shape[1]}  (image {h}x{w})")
    print(f"  n_classes : {lfw_people.target_names.shape[0]}  -> {list(lfw_people.target_names)}")
    return lfw_people


def compute_pca(X_train: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, components [n_components, n_features], singular values) via SVD of centred data.

    The right singular vectors ``V`` of the centred data matrix are the eigenvectors of the covariance
    matrix X^T X / (n-1), i.e. the principal components. Their singular values ``S`` relate to the
    eigenvalues by  lambda_i = S_i^2 / (n - 1).
    """
    mean = X_train.mean(axis=0)
    _, S, Vt = np.linalg.svd(X_train - mean, full_matrices=False)
    return mean, Vt[:n_components], S


def plot_gallery(images: np.ndarray, titles: list[str], h: int, w: int, path: Path, n_row: int = 3, n_col: int = 4) -> None:
    """Helper function to plot a gallery of portraits (as in the scikit-learn example)."""
    fig = plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    fig.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        ax = fig.add_subplot(n_row, n_col, i + 1)
        ax.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        ax.set_title(titles[i], size=12)
        ax.set_xticks(())
        ax.set_yticks(())
    save_figure(fig, path)


def plot_compactness(S: np.ndarray, n_train: int, n_components: int, path: Path) -> np.ndarray:
    """Cumulative explained-variance ratio of the first ``n_components`` eigenvalues."""
    explained_variance = (S**2) / (n_train - 1)
    ratio_cumsum = np.cumsum(explained_variance / explained_variance.sum())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(n_components), ratio_cumsum[:n_components])
    ax.set_xlabel("number of principal components")
    ax.set_ylabel("cumulative explained variance ratio")
    ax.set_title("Compactness")
    ax.grid(True, alpha=0.3)
    save_figure(fig, path)
    return ratio_cumsum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_components", type=int, default=150, help="size of the PCA face space")
    parser.add_argument("--n_estimators", type=int, default=150)
    parser.add_argument("--max_depth", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    ensure_dir(OUT_DIR)

    lfw_people = load_lfw()
    _, h, w = lfw_people.images.shape
    X, y, target_names = lfw_people.data, lfw_people.target, lfw_people.target_names
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # -- PCA / eigenfaces ---------------------------------------------------------------------
    mean, components, S = compute_pca(X_train, args.n_components)
    eigenfaces = components.reshape((args.n_components, h, w))
    X_train_pca = (X_train - mean) @ components.T  # project into face space
    X_test_pca = (X_test - mean) @ components.T  # NOTE: centred with the *training* mean
    print(f"\nface-space features: train {X_train_pca.shape}, test {X_test_pca.shape}")

    plot_gallery(eigenfaces, [f"eigenface {i}" for i in range(len(eigenfaces))], h, w, OUT_DIR / "eigenfaces.png")
    plot_gallery(np.concatenate([mean[None], X_train[:11]]), ["mean face"] + [target_names[t] for t in y_train[:11]], h, w, OUT_DIR / "mean_and_samples.png")
    ratio_cumsum = plot_compactness(S, len(X_train), args.n_components, OUT_DIR / "compactness.png")
    for k in (10, 50, 100, args.n_components):
        print(f"  first {k:>3d} components explain {100 * ratio_cumsum[k - 1]:.1f}% of the variance")

    # -- Random Forest on the PCA features ----------------------------------------------------
    estimator = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth, max_features=args.n_components, random_state=args.seed
    )
    estimator.fit(X_train_pca, y_train)  # expects X as [n_samples, n_features]
    predictions = estimator.predict(X_test_pca)
    correct = predictions == y_test
    accuracy = correct.mean()

    print(f"\nTotal testing : {len(X_test_pca)}")
    print(f"Total correct : {correct.sum()}")
    print(f"Accuracy      : {accuracy:.4f}")
    report = classification_report(y_test, predictions, target_names=target_names)
    print(report)

    plot_gallery(
        X_test[:12],
        [f"pred: {target_names[p].split()[-1]}\ntrue: {target_names[t].split()[-1]}" for p, t in zip(predictions[:12], y_test[:12])],
        h, w, OUT_DIR / "rf_predictions.png",
    )
    with open(OUT_DIR / "rf_results.json", "w") as f:
        json.dump({"accuracy": float(accuracy), "n_components": args.n_components, "report": report}, f, indent=2)
    print(f"[saved] {OUT_DIR / 'rf_results.json'}")


if __name__ == "__main__":
    main()
