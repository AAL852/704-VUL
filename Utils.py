"""
utils.py
--------
Shared evaluation utilities used across all model scripts.
Includes confusion matrix plotting, classification reporting,
training curve visualisation, and timing helpers.
"""

import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def elapsed(start: float) -> float:
    """Return seconds elapsed since start."""
    return time.time() - start


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_splits(data_dir: str = "data") -> dict:
    """
    Load preprocessed numpy arrays and metadata saved by 1_preprocess.py.

    Returns a dict with keys:
        X_train_padded, X_val_padded, X_test_padded,
        y_train, y_val, y_test,
        vocab, vocab_size, max_length
    """
    splits = {}
    for key in ["X_train_padded", "X_val_padded", "X_test_padded",
                "y_train", "y_val", "y_test"]:
        splits[key] = np.load(f"{data_dir}/{key}.npy", allow_pickle=True)

    with open(f"{data_dir}/vocab.pkl", "rb") as f:
        splits["vocab"] = pickle.load(f)

    with open(f"{data_dir}/split_metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    splits.update(meta)

    return splits


# ---------------------------------------------------------------------------
# Evaluation — deep learning models (Keras)
# ---------------------------------------------------------------------------
def evaluate_dl_model(model, X_train, y_train, X_test, y_test, train_time=None):
    """
    Evaluate a Keras model: prints classification report, plots confusion
    matrices for train and test sets, and reports inference times.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, X, y, label in [
        (axes[0], X_train, y_train, "Train"),
        (axes[1], X_test,  y_test,  "Test"),
    ]:
        start = time.process_time()
        preds = (model.predict(X) > 0.5).astype("int32")
        inference_time = time.process_time() - start

        cm = confusion_matrix(y, preds)
        sns.heatmap(cm, annot=True, fmt="g", ax=ax, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{label} — F1: {f1_score(y, preds):.4f} | "
                     f"Inference: {inference_time:.2f}s")
        print(f"\n{label} Classification Report:")
        print(classification_report(y, preds))

    if train_time:
        print(f"Training time: {train_time:.2f}s")

    plt.tight_layout()
    plt.savefig(f"outputs/{model.name}_confusion_matrices.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Evaluation — traditional ML models (sklearn)
# ---------------------------------------------------------------------------
def evaluate_ml_model(model, X_train, y_train, X_test, y_test,
                      train_time=None, model_name="model"):
    """
    Evaluate a fitted sklearn estimator: prints accuracy, classification
    report, timing, and plots normalised confusion matrices.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, X, y, label in [
        (axes[0], X_train, y_train, "Train"),
        (axes[1], X_test,  y_test,  "Test"),
    ]:
        start = time.time()
        preds = model.predict(X)
        inference_time = elapsed(start)

        ConfusionMatrixDisplay.from_predictions(
            y, preds, normalize="true", ax=ax, cmap="Blues"
        )
        acc = accuracy_score(y, preds)
        ax.set_title(f"{label} — Acc: {acc:.4f} | Inference: {inference_time:.2f}s")
        print(f"\n{label} Classification Report:")
        print(classification_report(y, preds))
        print(f"  Inference time: {inference_time:.2f}s")

    if train_time:
        print(f"Training time: {train_time:.2f}s")

    plt.tight_layout()
    plt.savefig(f"outputs/{model_name}_confusion_matrices.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Training curve plot (Keras history)
# ---------------------------------------------------------------------------
def plot_training_curves(history, model_name: str = "model"):
    """Plot loss and accuracy curves from a Keras training history object."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    ax1.plot(history.history["loss"],     label="Train Loss")
    ax1.plot(history.history["val_loss"], label="Val Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history.history["accuracy"],     label="Train Acc")
    ax2.plot(history.history["val_accuracy"], label="Val Acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.suptitle(f"{model_name} — Training Curves")
    plt.tight_layout()
    plt.savefig(f"outputs/{model_name}_training_curves.png", dpi=150)
    plt.close()
