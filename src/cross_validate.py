"""Stratified k-fold cross-validation - the headline, quotable numbers.

    python src/cross_validate.py

A single 20% hold-out contains only 31 bankrupt companies, so one test F1 has a
very wide confidence interval. Repeating the entire pipeline across k folds and
reporting mean +/- std is the defensible number to put in a report.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import CV_FOLDS, RANDOM_STATE, RESULTS_DIR, TOP_FEATURES, TRAIN_DATA_PATH
from data_preprocessing import load_data, split_features_target
from model_evaluation import metrics_to_row
from model_training import set_global_seeds
from pipeline import build_fold

REPORT_MODELS = ("gnb", "dnn", "ensemble", "ensemble@0.50")
REPORT_METRICS = ("accuracy", "balanced_accuracy", "precision", "recall",
                  "f1_score", "mcc", "roc_auc", "pr_auc")


def main():
    print("=" * 62)
    print(f"Company Bankruptcy Prediction - {CV_FOLDS}-fold cross-validation")
    print("=" * 62)

    df = load_data(TRAIN_DATA_PATH)
    X, y = split_features_target(df)
    print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} raw features, "
          f"{int(y.sum())} positive ({y.mean() * 100:.2f}%)")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    thresholds = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n--- Fold {fold_idx}/{CV_FOLDS} ---")
        set_global_seeds(RANDOM_STATE + fold_idx)

        fold = build_fold(
            X.iloc[train_idx], y.iloc[train_idx],
            X.iloc[test_idx], y.iloc[test_idx],
            top_features=TOP_FEATURES,
            random_state=RANDOM_STATE + fold_idx,
            verbose=False,
        )

        for key in REPORT_MODELS:
            row = metrics_to_row(fold["results"][key], key, fold["results"][key]["threshold"])
            row["fold"] = fold_idx
            rows.append(row)

        thresholds.append(fold["thresholds"]["ensemble"]["threshold"])
        ens = fold["results"]["ensemble"]
        print(f"  ensemble  thr={ens['threshold']:.2f}  "
              f"P={ens['precision']:.4f} R={ens['recall']:.4f} "
              f"F1={ens['f1_score']:.4f} AUC={ens['roc_auc']:.4f} PR-AUC={ens['pr_auc']:.4f}")

    per_fold = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    per_fold.to_csv(os.path.join(RESULTS_DIR, "cv_per_fold.csv"), index=False)

    agg = per_fold.groupby("model")[list(REPORT_METRICS)].agg(["mean", "std"])

    print("\n" + "=" * 62)
    print(f"CROSS-VALIDATED RESULTS ({CV_FOLDS} folds, mean +/- std)")
    print("=" * 62)
    for model in REPORT_MODELS:
        print(f"\n{model}:")
        for metric in REPORT_METRICS:
            mean = agg.loc[model, (metric, "mean")]
            std = agg.loc[model, (metric, "std")]
            print(f"  {metric:<19s} {mean:.4f} +/- {std:.4f}")

    print(f"\nTuned ensemble thresholds per fold: "
          f"{[f'{t:.2f}' for t in thresholds]}  (mean {np.mean(thresholds):.3f})")

    summary = {}
    for model in REPORT_MODELS:
        summary[model] = {
            metric: {
                "mean": float(agg.loc[model, (metric, "mean")]),
                "std": float(agg.loc[model, (metric, "std")]),
            }
            for metric in REPORT_METRICS
        }
    summary["_thresholds_per_fold"] = [float(t) for t in thresholds]
    summary["_n_folds"] = CV_FOLDS

    with open(os.path.join(RESULTS_DIR, "cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    flat = agg.copy()
    flat.columns = [f"{m}_{s}" for m, s in flat.columns]
    flat.to_csv(os.path.join(RESULTS_DIR, "cv_summary.csv"))

    print("\nSaved results/cv_per_fold.csv, results/cv_summary.{csv,json}")
    print("=" * 62)


if __name__ == "__main__":
    main()
