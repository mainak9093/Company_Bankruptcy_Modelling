"""Training pipeline: single stratified 80/20 hold-out, saves all artifacts.

    python src/train.py

Writes models/saved/{cleaner,scaler,gnb_model}.pkl, dnn_model.keras,
threshold.json, and results/holdout_results.json.
"""

import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    CLEANER_PATH,
    PROJECT_ROOT,
    DNN_MODEL_PATH,
    GNB_MODEL_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    SAVED_MODELS_DIR,
    SCALER_PATH,
    SELECTED_FEATURES_PATH,
    TEST_SIZE,
    THRESHOLD_PATH,
    TOP_FEATURES,
    TRAIN_DATA_PATH,
)
from data_preprocessing import check_data_quality, load_data, split_features_target
from model_evaluation import metrics_to_row, print_metrics
from model_training import set_global_seeds
from pipeline import build_fold


def main():
    set_global_seeds(RANDOM_STATE)

    print("=" * 62)
    print("Company Bankruptcy Prediction - Training (80/20 hold-out)")
    print("=" * 62)

    print("\n[1/5] Loading data...")
    df = load_data(TRAIN_DATA_PATH)
    print(f"  Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    check_data_quality(df)

    X, y = split_features_target(df)
    print(f"  Class distribution: {y.value_counts().to_dict()} "
          f"({y.mean() * 100:.2f}% positive)")

    print("\n[2/5] Splitting off the held-out test set...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  train+val: {X_train_full.shape[0]} rows ({int(y_train_full.sum())} positive)")
    print(f"  test     : {X_test.shape[0]} rows ({int(y_test.sum())} positive)")

    print("\n[3/5] Fitting the pipeline (clean -> ANOVA -> scale -> SMOTE -> train)...")
    fold = build_fold(
        X_train_full, y_train_full, X_test, y_test,
        top_features=TOP_FEATURES, random_state=RANDOM_STATE, verbose=True,
    )

    print("\n[4/5] Results on the held-out test set")
    print("\nTop 10 features by ANOVA F-score:")
    print(fold["feature_scores"].head(10).to_string(index=False))

    for key, label in (
        ("gnb", "GaussianNB (alone)"),
        ("dnn", "DNN (alone)"),
        ("ensemble", "Ensemble DNN + GaussianNB"),
        ("ensemble@0.50", "Ensemble at default threshold 0.50"),
    ):
        print_metrics(fold["results"][key], label, fold["results"][key]["threshold"])

    print("\n[5/5] Saving artifacts...")
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(CLEANER_PATH, "wb") as f:
        pickle.dump(fold["cleaner"], f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(fold["scaler"], f)
    with open(GNB_MODEL_PATH, "wb") as f:
        pickle.dump(fold["gnb_model"], f)
    fold["dnn_model"].save(DNN_MODEL_PATH)

    with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
        json.dump(fold["thresholds"], f, indent=2)

    pd.DataFrame({"feature": fold["selected_features"]}).to_csv(
        SELECTED_FEATURES_PATH, index=False
    )
    fold["feature_scores"].to_csv(
        os.path.join(PROCESSED_DATA_DIR, "anova_feature_scores.csv"), index=False
    )

    summary_rows = [
        metrics_to_row(fold["results"][k], k, fold["results"][k]["threshold"])
        for k in ("gnb", "dnn", "ensemble", "ensemble@0.50")
    ]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(RESULTS_DIR, "holdout_results.csv"), index=False)

    with open(os.path.join(RESULTS_DIR, "holdout_results.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "protocol": {
                    "test_size": TEST_SIZE,
                    "random_state": RANDOM_STATE,
                    "top_features": TOP_FEATURES,
                    "threshold_tuned_on": "validation split (never the test set)",
                },
                "splits": fold["splits"],
                "selected_features": fold["selected_features"],
                "cleaner": {
                    "constant_dropped": fold["cleaner"].constant_columns_,
                    "high_error_dropped": fold["cleaner"].high_error_columns_,
                    "low_error_repaired": fold["cleaner"].low_error_columns_,
                    "correlated_dropped": fold["cleaner"].correlated_columns_,
                    "n_features_after_cleaning": len(fold["cleaner"].feature_names_),
                },
                "dnn_fit": fold["fit_info"],
                "metrics": summary_rows,
            },
            f,
            indent=2,
        )

    for path in (CLEANER_PATH, SCALER_PATH, GNB_MODEL_PATH, DNN_MODEL_PATH,
                 THRESHOLD_PATH, SELECTED_FEATURES_PATH):
        print(f"  saved {os.path.relpath(path, PROJECT_ROOT).replace(os.sep, '/')}")
    print("  saved results/holdout_results.{csv,json}")

    print("\n" + "=" * 62)
    print("SUMMARY (held-out test set, threshold tuned on validation)")
    print("=" * 62)
    cols = ["model", "threshold", "accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("=" * 62)


if __name__ == "__main__":
    main()
