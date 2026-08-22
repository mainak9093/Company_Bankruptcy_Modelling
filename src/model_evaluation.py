"""Evaluation metrics and decision-threshold tuning."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_optimal_threshold(y_true, y_pred_probs, start=0.01, end=1.00, step=0.01):
    """Find the threshold maximising F1 on the data supplied.

    Call this with a VALIDATION set. Tuning the threshold on the test set and
    then reporting the resulting F1 on that same test set - as the original
    ``train.py`` did - reports an optimistically biased number.
    """
    thresholds = np.arange(start, end, step)
    # Starting at 0.0 rather than -inf means a run where no threshold separates
    # the classes falls back to 0.5 instead of returning the first threshold
    # tried, which would classify everything as positive.
    best_f1 = 0.0
    best_threshold = 0.5
    results = []

    for threshold in thresholds:
        y_pred = (y_pred_probs >= threshold).astype(int)
        current_f1 = f1_score(y_true, y_pred, zero_division=0)
        results.append({"threshold": float(threshold), "f1_score": float(current_f1)})
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(threshold)

    return best_threshold, float(best_f1), results


def evaluate_model(y_true, y_pred, y_pred_probs=None):
    """Compute the full metric set for a binary classifier."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "confusion_matrix": cm,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1],
            target_names=["Non-bankrupt", "Bankrupt"],
            zero_division=0, digits=4,
        ),
    }

    if y_pred_probs is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_probs))
        # PR-AUC is the honest headline metric for a 2.8% positive rate.
        metrics["pr_auc"] = float(average_precision_score(y_true, y_pred_probs))

    return metrics


def print_metrics(metrics, model_name="Model", threshold=None):
    """Print a metric dictionary in a readable block."""
    print(f"\n{'=' * 62}")
    title = f"{model_name} - Evaluation"
    if threshold is not None:
        title += f" (threshold = {threshold:.2f})"
    print(title)
    print("=" * 62)
    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"Precision         : {metrics['precision']:.4f}")
    print(f"Recall            : {metrics['recall']:.4f}")
    print(f"F1-Score          : {metrics['f1_score']:.4f}")
    print(f"MCC               : {metrics['mcc']:.4f}")
    if "roc_auc" in metrics:
        print(f"ROC-AUC           : {metrics['roc_auc']:.4f}")
    if "pr_auc" in metrics:
        print(f"PR-AUC            : {metrics['pr_auc']:.4f}")
    print("\nConfusion Matrix (rows = true 0/1, cols = pred 0/1):")
    print(metrics["confusion_matrix"])
    print(f"\n{metrics['classification_report']}")
    print("=" * 62)


def metrics_to_row(metrics, name, threshold=None):
    """Flatten a metric dictionary into a row for a summary table."""
    row = {
        "model": name,
        "threshold": threshold,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "mcc": metrics["mcc"],
        "roc_auc": metrics.get("roc_auc"),
        "pr_auc": metrics.get("pr_auc"),
        "tn": metrics["tn"], "fp": metrics["fp"],
        "fn": metrics["fn"], "tp": metrics["tp"],
    }
    return row
