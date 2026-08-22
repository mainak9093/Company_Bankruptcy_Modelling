"""Inference on new data using the saved artifacts.

    python src/predict.py <input.csv> [output.csv]

The input CSV must contain the same raw feature columns as Train.csv. The
label column is optional; if present it is ignored for prediction (use
evaluate.py to score labelled data).
"""

import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from tensorflow.keras.models import load_model

from config import (
    CLEANER_PATH,
    DNN_MODEL_PATH,
    ENSEMBLE_THRESHOLD,
    GNB_MODEL_PATH,
    RESULTS_DIR,
    SCALER_PATH,
    SELECTED_FEATURES_PATH,
    TARGET_COLUMN,
    THRESHOLD_PATH,
)
from data_preprocessing import load_data
from feature_selection import get_selected_features_subset
from model_training import get_ensemble_predictions


def load_artifacts():
    """Load the cleaner, scaler, both models, the feature list and the threshold."""
    required = [CLEANER_PATH, SCALER_PATH, GNB_MODEL_PATH, DNN_MODEL_PATH, SELECTED_FEATURES_PATH]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing artifacts:\n  "
            + "\n  ".join(missing)
            + "\n\nRun `python src/train.py` first."
        )

    with open(CLEANER_PATH, "rb") as f:
        cleaner = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(GNB_MODEL_PATH, "rb") as f:
        gnb_model = pickle.load(f)
    dnn_model = load_model(DNN_MODEL_PATH, compile=False)

    selected_features = pd.read_csv(SELECTED_FEATURES_PATH)["feature"].tolist()

    threshold = ENSEMBLE_THRESHOLD
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, encoding="utf-8") as f:
            threshold = json.load(f)["ensemble"]["threshold"]

    return cleaner, scaler, dnn_model, gnb_model, selected_features, threshold


def prepare_features(data, cleaner, scaler, selected_features):
    """Apply the exact training-time transform chain to new data.

    The original predict.py skipped cleaning entirely and only sub-selected
    columns, so production inputs were transformed differently from training
    inputs.
    """
    X = data.drop(columns=[TARGET_COLUMN]) if TARGET_COLUMN in data.columns else data
    X_clean = cleaner.transform(X)
    X_selected = get_selected_features_subset(X_clean, selected_features)
    return scaler.transform(X_selected)


def predict(data_path, output_path=None):
    """Score a CSV and return a dataframe of probabilities and predictions."""
    print("=" * 62)
    print("Company Bankruptcy Prediction - Inference")
    print("=" * 62)

    cleaner, scaler, dnn_model, gnb_model, selected_features, threshold = load_artifacts()
    print(f"  Artifacts loaded ({len(selected_features)} features, threshold {threshold:.2f})")

    data = load_data(data_path)
    print(f"  Input: {data.shape[0]} rows x {data.shape[1]} columns")

    X_scaled = prepare_features(data, cleaner, scaler, selected_features)
    ensemble_probs, dnn_probs, gnb_probs = get_ensemble_predictions(dnn_model, gnb_model, X_scaled)
    predictions = (ensemble_probs >= threshold).astype(int)

    results_df = pd.DataFrame(
        {
            "DNN_Probability": dnn_probs,
            "GNB_Probability": gnb_probs,
            "Ensemble_Probability": ensemble_probs,
            "Prediction": predictions,
            "Bankruptcy_Risk": ["High" if p == 1 else "Low" for p in predictions],
        }
    )
    if TARGET_COLUMN in data.columns:
        results_df.insert(0, "True_Label", data[TARGET_COLUMN].values)

    print(f"\n  High risk (predicted bankrupt) : {int((predictions == 1).sum())}")
    print(f"  Low risk (predicted solvent)   : {int((predictions == 0).sum())}")
    print(f"  Mean ensemble probability      : {ensemble_probs.mean():.4f}")

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\n  Saved to {output_path}")

    print("=" * 62)
    return results_df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <data_path> [output_path]")
        print("Example: python src/predict.py data/raw/sample.csv results/predictions.csv")
        sys.exit(1)

    data_path = sys.argv[1]
    if not os.path.exists(data_path):
        print(f"Error: file not found - {data_path}")
        sys.exit(1)

    output_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RESULTS_DIR, "predictions.csv")
    predict(data_path, output_path)
