"""Inference script for making predictions on new data."""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    SCALER_PATH,
    GNB_MODEL_PATH,
    DNN_MODEL_PATH,
    ENSEMBLE_THRESHOLD,
)
from feature_selection import get_selected_features_subset


def load_models_and_scaler():
    """Load all saved models and scaler."""
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    with open(GNB_MODEL_PATH, "rb") as f:
        gnb_model = pickle.load(f)

    dnn_model = load_model(DNN_MODEL_PATH)

    return dnn_model, gnb_model, scaler


def load_selected_features():
    """Load selected features used during training."""
    processed_dir = os.path.dirname(SCALER_PATH).replace("saved", "")
    features_path = os.path.join(processed_dir, "processed", "selected_features.csv")

    if os.path.exists(features_path):
        df = pd.read_csv(features_path)
        return df["feature"].tolist()
    else:
        raise FileNotFoundError(f"Selected features file not found: {features_path}")


def predict(data_path, output_path=None):
    """
    Make predictions on new data.

    Args:
        data_path: Path to CSV file with new data
        output_path: Path to save predictions (optional)

    Returns:
        DataFrame with predictions
    """
    print("\n" + "="*60)
    print("Loading Models and Data")
    print("="*60)

    dnn_model, gnb_model, scaler = load_models_and_scaler()
    print("✓ Models and scaler loaded")

    selected_features = load_selected_features()
    print(f"✓ Using {len(selected_features)} selected features")

    data = pd.read_csv(data_path)
    print(f"✓ Loaded data with shape: {data.shape}")

    print("\n" + "="*60)
    print("Preparing Data for Prediction")
    print("="*60)

    X = get_selected_features_subset(data, selected_features)
    X_scaled = scaler.transform(X)
    print(f"✓ Data scaled and features selected")

    print("\n" + "="*60)
    print("Making Predictions")
    print("="*60)

    dnn_probs = dnn_model.predict(X_scaled, verbose=0).flatten()
    gnb_probs = gnb_model.predict_proba(X_scaled)[:, 1]
    ensemble_probs = (dnn_probs + gnb_probs) / 2

    predictions = (ensemble_probs > ENSEMBLE_THRESHOLD).astype(int)

    results_df = pd.DataFrame(
        {
            "DNN_Probability": dnn_probs,
            "GNB_Probability": gnb_probs,
            "Ensemble_Probability": ensemble_probs,
            "Prediction": predictions,
            "Bankruptcy_Risk": ["High" if p == 1 else "Low" for p in predictions],
        }
    )

    print(f"\nPrediction Summary:")
    print(f"High Risk (Bankrupt): {(predictions == 1).sum()}")
    print(f"Low Risk (Non-Bankrupt): {(predictions == 0).sum()}")
    print(f"Average Bankruptcy Probability: {ensemble_probs.mean():.4f}")

    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ Predictions saved to: {output_path}")
        print("="*60 + "\n")
        return results_df

    print("="*60 + "\n")
    return results_df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <data_path> [output_path]")
        print("Example: python predict.py sample.csv predictions.csv")
        sys.exit(1)

    data_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(data_path):
        print(f"Error: File not found - {data_path}")
        sys.exit(1)

    predict(data_path, output_path)
