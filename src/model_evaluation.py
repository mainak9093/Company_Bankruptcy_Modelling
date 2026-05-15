"""Model evaluation and threshold optimization utilities."""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)


def find_optimal_threshold(y_true, y_pred_probs, start=0.30, end=0.60, step=0.01):
    """Find optimal threshold that maximizes F1-score."""
    thresholds = np.arange(start, end, step)
    best_f1 = 0
    best_threshold = 0.5
    results = []

    for threshold in thresholds:
        y_pred = (y_pred_probs > threshold).astype(int)
        current_f1 = f1_score(y_true, y_pred, zero_division=0)
        results.append({"threshold": threshold, "f1_score": current_f1})

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    return best_threshold, best_f1, results


def evaluate_model(y_true, y_pred, y_pred_probs=None):
    """Evaluate model and return comprehensive metrics."""
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)

    metrics = {
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": report,
    }

    if y_pred_probs is not None:
        roc_auc = roc_auc_score(y_true, y_pred_probs)
        metrics["roc_auc"] = roc_auc

    return metrics


def print_metrics(metrics, model_name="Model"):
    """Print evaluation metrics in a formatted way."""
    print(f"\n{'='*50}")
    print(f"{model_name} Evaluation Results")
    print(f"{'='*50}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    if "roc_auc" in metrics:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")
    print(f"\n{metrics['classification_report']}")
    print(f"{'='*50}\n")
