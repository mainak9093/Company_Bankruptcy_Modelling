"""Baseline comparison under the identical leak-free protocol.

    python src/baselines.py

The report claims the DNN+GNB ensemble beats simpler models. That claim is only
meaningful if every model is measured the same way, so each baseline here gets
the same cleaning, ANOVA selection, scaling, SMOTE, validation-tuned threshold
and held-out test set as the ensemble.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from config import (
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_SIZE,
    TOP_FEATURES,
    TRAIN_DATA_PATH,
    USE_SMOTE,
    VAL_SIZE,
)
from data_preprocessing import DataCleaner, load_data, split_features_target
from feature_selection import get_selected_features_subset, select_features_anova
from model_evaluation import evaluate_model, find_optimal_threshold, metrics_to_row
from model_training import apply_smote, scale_features, set_global_seeds


def get_baselines():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "GaussianNB": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=RANDOM_STATE
        ),
    }


def main():
    set_global_seeds(RANDOM_STATE)

    print("=" * 62)
    print("Baseline comparison (same protocol as the ensemble)")
    print("=" * 62)

    df = load_data(TRAIN_DATA_PATH)
    X, y = split_features_target(df)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_full, y_train_full, test_size=VAL_SIZE,
        stratify=y_train_full, random_state=RANDOM_STATE,
    )

    cleaner = DataCleaner(verbose=False)
    X_fit_c = cleaner.fit_transform(X_fit, y_fit)
    X_val_c = cleaner.transform(X_val)
    X_test_c = cleaner.transform(X_test)

    selected, _ = select_features_anova(X_fit_c, y_fit, top_n=TOP_FEATURES)
    X_fit_s = get_selected_features_subset(X_fit_c, selected)
    X_val_s = get_selected_features_subset(X_val_c, selected)
    X_test_s = get_selected_features_subset(X_test_c, selected)

    X_fit_sc, X_val_sc, X_test_sc, _ = scale_features(X_fit_s, X_val_s, X_test_s)

    if USE_SMOTE:
        X_bal, y_bal = apply_smote(X_fit_sc, y_fit, verbose=False)
    else:
        X_bal, y_bal = X_fit_sc, np.asarray(y_fit)

    print(f"train {X_bal.shape}  val {X_val_sc.shape} (pos {int(y_val.sum())})  "
          f"test {X_test_sc.shape} (pos {int(y_test.sum())})\n")

    rows = []
    for name, model in get_baselines().items():
        model.fit(X_bal, y_bal)
        val_probs = model.predict_proba(X_val_sc)[:, 1]
        thr, val_f1, _ = find_optimal_threshold(np.asarray(y_val), val_probs)

        test_probs = model.predict_proba(X_test_sc)[:, 1]
        metrics = evaluate_model(np.asarray(y_test), (test_probs >= thr).astype(int), test_probs)
        rows.append(metrics_to_row(metrics, name, thr))
        print(f"{name:<22s} thr={thr:.2f}  P={metrics['precision']:.4f} "
              f"R={metrics['recall']:.4f} F1={metrics['f1_score']:.4f} "
              f"AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f}")

    summary = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary.to_csv(os.path.join(RESULTS_DIR, "baseline_results.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "baseline_results.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print("\n" + "=" * 62)
    cols = ["model", "threshold", "accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("=" * 62)
    print("\nSaved results/baseline_results.{csv,json}")


if __name__ == "__main__":
    main()
