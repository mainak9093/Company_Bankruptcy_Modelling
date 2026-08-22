"""Score the saved models against a labelled CSV.

    python src/evaluate.py <labelled.csv>

Use this on genuinely unseen labelled data. Running it on Train.csv scores the
models on rows they were fitted on and will look far better than reality - the
number to quote in a report comes from train.py or cross_validate.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from config import TARGET_COLUMN, TRAIN_DATA_PATH
from data_preprocessing import load_data
from model_evaluation import evaluate_model, print_metrics
from model_training import get_ensemble_predictions
from predict import load_artifacts, prepare_features


def main(data_path):
    cleaner, scaler, dnn_model, gnb_model, selected_features, threshold = load_artifacts()

    data = load_data(data_path)
    if TARGET_COLUMN not in data.columns:
        print(f"Error: {data_path} has no '{TARGET_COLUMN}' column, so it cannot be scored.")
        sys.exit(1)

    y_true = data[TARGET_COLUMN].values
    if len(np.unique(y_true)) < 2:
        print(f"Warning: only one class present ({np.unique(y_true)}); "
              "precision/recall/AUC are undefined.")

    if os.path.abspath(data_path) == os.path.abspath(TRAIN_DATA_PATH):
        print("\n*** WARNING: scoring on Train.csv, which the models were fitted on. ***")
        print("*** These numbers are optimistic. Quote train.py / cross_validate.py. ***\n")

    X_scaled = prepare_features(data, cleaner, scaler, selected_features)
    ens, dnn, gnb = get_ensemble_predictions(dnn_model, gnb_model, X_scaled)

    print(f"Rows: {len(y_true)}  Positives: {int(np.sum(y_true))}  Threshold: {threshold:.2f}")
    for name, probs in (("GaussianNB", gnb), ("DNN", dnn), ("Ensemble", ens)):
        metrics = evaluate_model(y_true, (probs >= threshold).astype(int),
                                 probs if len(np.unique(y_true)) > 1 else None)
        print_metrics(metrics, name, threshold)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/evaluate.py <labelled_csv>")
        sys.exit(1)
    if not os.path.exists(sys.argv[1]):
        print(f"Error: file not found - {sys.argv[1]}")
        sys.exit(1)
    main(sys.argv[1])
