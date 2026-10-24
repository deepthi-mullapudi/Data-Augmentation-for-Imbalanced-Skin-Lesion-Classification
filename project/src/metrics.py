"""Classification, image-quality, and nearest-neighbour metrics."""

"""Classifier evaluation metrics and confusion-matrix export."""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    recall_score, roc_auc_score,
)

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        logits_all.append(logits.detach().cpu())
        labels_all.append(labels.detach().cpu())
    if not logits_all:
        raise ValueError("Evaluation loader is empty.")
    return torch.cat(logits_all).numpy(), torch.cat(labels_all).numpy()


def calculate_metrics(y_true, logits):
    """Return metrics from true integer labels and raw class logits."""
    y_true = np.asarray(y_true, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != len(CLASS_NAMES):
        raise ValueError(f"Expected logits with shape (N, 7), got {logits.shape}")
    if len(y_true) != len(logits):
        raise ValueError("Number of labels and logits differs.")
    if set(np.unique(y_true)) != set(range(len(CLASS_NAMES))):
        raise ValueError("Evaluation labels must contain all seven classes.")

    # Stable softmax for one-vs-rest ROC-AUC.
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)

    recalls = recall_score(
        y_true, predictions, labels=list(range(7)), average=None, zero_division=0
    )
    one_hot = np.eye(7, dtype=np.int64)[y_true]
    auc = roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr")
    cm = confusion_matrix(y_true, predictions, labels=list(range(7)))
    result = {
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "bal_acc": float(balanced_accuracy_score(y_true, predictions)),
        "auc": float(auc),
    }
    result.update({f"recall_{name}": float(recalls[i]) for i, name in enumerate(CLASS_NAMES)})
    return result, cm, predictions


def save_confusion_matrix(cm, output_dir, prefix="test"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{prefix}_confusion_matrix.csv"
    png_path = output_dir / f"{prefix}_confusion_matrix.png"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *CLASS_NAMES])
        for name, row in zip(CLASS_NAMES, cm):
            writer.writerow([name, *row.tolist()])

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(7), yticks=np.arange(7),
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        xlabel="Predicted label", ylabel="True label",
        title=f"{prefix.capitalize()} confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return csv_path, png_path
