"""Generate the figures used by docs/reports/src/EE708_report.tex.

    python src/make_report_figures.py

Every figure is drawn from the same leak-free pipeline the metrics come from,
so the report cannot drift from the code. Outputs go to
docs/reports/src/figures/ as vector PDFs.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split

from config import (
    PROJECT_ROOT,
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_SIZE,
    TOP_FEATURES,
    TRAIN_DATA_PATH,
)
from data_preprocessing import load_data, split_features_target
from model_training import set_global_seeds
from pipeline import build_fold

FIG_DIR = os.path.join(PROJECT_ROOT, "docs", "reports", "src", "figures")

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, PROJECT_ROOT)}")


def fig_class_balance(y):
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    counts = y.value_counts().sort_index()
    bars = ax.bar(["Non-bankrupt (0)", "Bankrupt (1)"], counts.values,
                  color=["#4C72B0", "#C44E52"], width=0.55)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 90, f"{v:,}",
                ha="center", fontsize=9)
    ax.set_ylabel("Companies")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_title(f"Class distribution ({counts[1] / counts.sum() * 100:.2f}% positive)")
    save(fig, "class_balance.pdf")


def fig_correlation_heatmap(X_clean):
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    corr = X_clean.corr()
    sns.heatmap(corr, mask=np.eye(len(corr), dtype=bool), cmap="RdBu_r",
                vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws={"shrink": 0.7, "label": "Pearson r"},
                xticklabels=False, yticklabels=False)
    ax.set_title(f"Feature correlation after cleaning ({corr.shape[0]} features)")
    save(fig, "correlation_heatmap.pdf")


def fig_anova_scores(feature_scores, top_n=15):
    top = feature_scores.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.barh(range(len(top)), top["f_score"], color="#4C72B0", height=0.7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f[:44] for f in top["feature"]], fontsize=7.5)
    ax.set_xlabel("ANOVA F-score")
    ax.set_title(f"Top {top_n} features by ANOVA F-test")
    save(fig, "anova_scores.pdf")


def fig_confusion_matrix(metrics):
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(3.3, 2.8))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                annot_kws={"size": 12},
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    ax.set_title(f"Ensemble, threshold {metrics['threshold']:.2f}")
    save(fig, "confusion_matrix.pdf")


def fig_roc_pr(y_true, probs, roc_auc, pr_auc, prevalence):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))

    fpr, tpr, _ = roc_curve(y_true, probs)
    axes[0].plot(fpr, tpr, color="#4C72B0", lw=1.8, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1, label="chance")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    prec, rec, _ = precision_recall_curve(y_true, probs)
    axes[1].plot(rec, prec, color="#C44E52", lw=1.8, label=f"AP = {pr_auc:.3f}")
    axes[1].axhline(prevalence, ls="--", color="grey", lw=1,
                    label=f"chance = {prevalence:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curve")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    save(fig, "roc_pr_curves.pdf")


def fig_threshold_sweep(y_true, probs):
    from sklearn.metrics import f1_score, precision_score, recall_score

    ts = np.arange(0.01, 1.00, 0.01)
    f1 = [f1_score(y_true, (probs >= t).astype(int), zero_division=0) for t in ts]
    pr = [precision_score(y_true, (probs >= t).astype(int), zero_division=0) for t in ts]
    rc = [recall_score(y_true, (probs >= t).astype(int), zero_division=0) for t in ts]

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.plot(ts, f1, color="#4C72B0", lw=1.8, label="F1")
    ax.plot(ts, pr, color="#C44E52", lw=1.2, ls="--", label="Precision")
    ax.plot(ts, rc, color="#55A868", lw=1.2, ls=":", label="Recall")
    ax.axvspan(0.30, 0.60, color="grey", alpha=0.14)
    ax.text(0.45, 0.93, "original\nsearch window", ha="center", fontsize=7, color="#555")
    best = ts[int(np.argmax(f1))]
    ax.axvline(best, color="black", lw=1, ls="-.")
    ax.text(best + 0.015, 0.55, f"optimum\n{best:.2f}", fontsize=7.5)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title("Threshold sweep on the test set")
    ax.legend(frameon=False, fontsize=8, loc="center left")
    save(fig, "threshold_sweep.pdf")


def fig_cv_spread():
    path = os.path.join(RESULTS_DIR, "cv_per_fold.csv")
    if not os.path.exists(path):
        print("  (skipping CV figure: run src/cross_validate.py first)")
        return
    df = pd.read_csv(path)
    df = df[df["model"].isin(["gnb", "dnn", "ensemble"])]
    labels = {"gnb": "GaussianNB", "dnn": "DNN", "ensemble": "Ensemble"}

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for ax, metric, title in zip(axes, ["f1_score", "roc_auc"], ["F1-score", "ROC-AUC"]):
        data = [df[df["model"] == m][metric].values for m in labels]
        ax.boxplot(data, tick_labels=list(labels.values()), widths=0.5)
        for i, vals in enumerate(data, start=1):
            ax.scatter(np.full(len(vals), i), vals, s=14, color="#C44E52", zorder=3)
        ax.set_title(f"{title} across 5 folds")
        ax.set_ylabel(title)
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    save(fig, "cv_spread.pdf")


def main():
    set_global_seeds(RANDOM_STATE)
    print("Generating report figures...")

    df = load_data(TRAIN_DATA_PATH)
    X, y = split_features_target(df)

    fig_class_balance(y)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    fold = build_fold(X_train_full, y_train_full, X_test, y_test,
                      top_features=TOP_FEATURES, random_state=RANDOM_STATE, verbose=False)

    # Both figures must come from the pipeline's OWN fitted cleaner and ANOVA
    # ranking. Re-deriving them here from a different split silently produces a
    # different surviving feature set (63 vs 65) and a different top-15 order,
    # which then contradicts the tables in the report.
    fig_correlation_heatmap(fold["X_fit_clean"])
    fig_anova_scores(fold["feature_scores"])

    ens = fold["results"]["ensemble"]
    probs = fold["probs"]["test_ensemble"]
    y_test_arr = np.asarray(y_test)

    fig_confusion_matrix(ens)
    fig_roc_pr(y_test_arr, probs, ens["roc_auc"], ens["pr_auc"], float(y_test_arr.mean()))
    fig_threshold_sweep(y_test_arr, probs)
    fig_cv_spread()

    # Numbers the .tex quotes, dumped so the two can be diffed if they drift.
    os.makedirs(FIG_DIR, exist_ok=True)
    with open(os.path.join(FIG_DIR, "report_numbers.json"), "w", encoding="utf-8") as f:
        json.dump({k: {kk: vv for kk, vv in v.items()
                       if kk not in ("confusion_matrix", "classification_report")}
                   for k, v in fold["results"].items()}, f, indent=2)
    print(f"  wrote {os.path.relpath(os.path.join(FIG_DIR, 'report_numbers.json'), PROJECT_ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
