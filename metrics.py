"""
metrics.py
==========
Section 6 (statistical logging half): full post-training evaluation --
overall accuracy, per-class precision/recall/specificity/F1, a confusion
matrix heatmap (Seaborn, high-res PNG), and per-class ROC-AUC curves.
"""

import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for saving figures on servers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader


@torch.no_grad()
def run_inference(model: torch.nn.Module, loader: DataLoader, device: str):
    """Runs the model over a DataLoader and collects predictions/probabilities/labels."""
    model.eval()
    model.to(device)

    all_labels, all_preds, all_probs = [], [], []
    for images, labels, _ in loader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict:
    """
    Computes precision, recall (sensitivity), specificity, and F1 per
    class from the confusion matrix directly (one-vs-rest), so the
    definitions are transparent and don't depend on sklearn averaging
    quirks.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    results = {}
    total = cm.sum()

    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0          # sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        results[name] = {
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1,
            "support": int(cm[i, :].sum()),
        }
    return results


def overall_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: str, dpi: int = 300):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Retinal-SwinNet -- Confusion Matrix (Test Set)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def plot_roc_curves(y_true, y_probs, class_names, save_path: str, dpi: int = 300) -> Dict[str, float]:
    """One-vs-rest ROC-AUC curves, one line per class, saved to a single PNG."""
    plt.figure(figsize=(8, 6))
    auc_scores = {}

    for i, name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        y_score = y_probs[:, i]

        if len(np.unique(y_true_binary)) < 2:
            # Can't compute ROC-AUC for a class with no positive (or no
            # negative) examples in this split -- skip gracefully.
            auc_scores[name] = float("nan")
            continue

        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        auc = roc_auc_score(y_true_binary, y_score)
        auc_scores[name] = float(auc)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Retinal-SwinNet -- Per-Class ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.close()

    return auc_scores


def full_evaluation_report(
    model: torch.nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    device: str,
    output_dir: str,
) -> Dict:
    """
    Runs inference on the test set and produces:
      - outputs/confusion_matrix.png
      - outputs/roc_curves.png
      - a dict of all numeric metrics (also useful for the paper generator)
    """
    y_true, y_pred, y_probs = run_inference(model, test_loader, device)

    acc = overall_accuracy(y_true, y_pred)
    per_class = per_class_metrics(y_true, y_pred, class_names)

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    roc_path = os.path.join(output_dir, "roc_curves.png")

    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
    auc_scores = plot_roc_curves(y_true, y_probs, class_names, roc_path)

    report = {
        "overall_test_accuracy": acc,
        "per_class_metrics": per_class,
        "per_class_roc_auc": auc_scores,
        "macro_f1": float(np.mean([v["f1_score"] for v in per_class.values()])),
        "confusion_matrix_path": cm_path,
        "roc_curve_path": roc_path,
    }

    print(f"\nOverall Top-1 Test Accuracy: {acc * 100:.2f}%")
    print(f"{'Class':<24}{'Precision':>10}{'Recall':>10}{'Specif.':>10}{'F1':>10}{'AUC':>10}")
    for name in class_names:
        m = per_class[name]
        auc = auc_scores[name]
        auc_str = f"{auc:.3f}" if not np.isnan(auc) else "n/a"
        print(f"{name:<24}{m['precision']:>10.3f}{m['recall']:>10.3f}"
              f"{m['specificity']:>10.3f}{m['f1_score']:>10.3f}{auc_str:>10}")

    return report
