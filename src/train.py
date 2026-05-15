"""Main training pipeline for bankruptcy prediction model."""

import os
import sys
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    TRAIN_DATA_PATH,
    PROCESSED_DATA_DIR,
    SAVED_MODELS_DIR,
    TEST_SIZE,
    TOP_FEATURES,
    RANDOM_STATE,
    ENSEMBLE_THRESHOLD,
    SCALER_PATH,
    GNB_MODEL_PATH,
    DNN_MODEL_PATH,
)
from data_preprocessing import load_data, preprocess_data
from feature_selection import select_features_anova, get_selected_features_subset
from model_training import (
    apply_smote,
    scale_features,
    calculate_class_weights,
    train_dnn,
    train_gaussian_nb,
    get_ensemble_predictions,
)
from model_evaluation import find_optimal_threshold, evaluate_model, print_metrics


def main():
    """Execute complete training pipeline."""
    print("="*60)
    print("Company Bankruptcy Prediction - Training Pipeline")
    print("="*60)

    print("\n[1/6] Loading data...")
    df = load_data(TRAIN_DATA_PATH)
    print(f"Loaded data shape: {df.shape}")

    print("\n[2/6] Preprocessing data...")
    df = preprocess_data(df)

    print("\n[3/6] Selecting features...")
    X = df.drop(columns=["Bankrupt?"])
    y = df["Bankrupt?"]
    selected_features, feature_scores = select_features_anova(X, y, top_n=TOP_FEATURES)
    print(f"Selected {len(selected_features)} features")
    print("\nTop 10 features by ANOVA score:")
    print(feature_scores.head(10).to_string())

    X_selected = get_selected_features_subset(X, selected_features)

    print("\n[4/6] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Class distribution in train: {y_train.value_counts().to_dict()}")

    print("\n[5/6] Applying SMOTE and scaling...")
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_sm, X_test)

    print("\n[6/6] Training models...")
    class_weights = calculate_class_weights(y_train_sm)
    print(f"Class weights: {class_weights}")

    # Split training data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_scaled, y_train_sm, test_size=0.2, random_state=RANDOM_STATE
    )

    print("\nTraining DNN...")
    dnn_model, dnn_history = train_dnn(
        X_train_split, y_train_split, X_val, y_val, class_weights
    )
    print("✓ DNN training complete")

    print("\nTraining Gaussian Naive Bayes...")
    gnb_model = train_gaussian_nb(X_train_scaled, y_train_sm)
    print("✓ GNB training complete")

    print("\nGenerating ensemble predictions...")
    ensemble_probs, dnn_probs, gnb_probs = get_ensemble_predictions(
        dnn_model, gnb_model, X_test_scaled
    )

    print("\nOptimizing threshold...")
    best_threshold, best_f1, threshold_results = find_optimal_threshold(
        y_test, ensemble_probs
    )
    print(f"Optimal threshold: {best_threshold:.2f}, F1-Score: {best_f1:.4f}")

    y_pred = (ensemble_probs > best_threshold).astype(int)

    print("\nEvaluating ensemble model...")
    metrics = evaluate_model(y_test, y_pred, ensemble_probs)
    print_metrics(metrics, model_name="Ensemble (DNN + GaussianNB)")

    print("\nSaving models and artifacts...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to {SCALER_PATH}")

    with open(GNB_MODEL_PATH, "wb") as f:
        pickle.dump(gnb_model, f)
    print(f"✓ GNB model saved to {GNB_MODEL_PATH}")

    dnn_model.save(DNN_MODEL_PATH)
    print(f"✓ DNN model saved to {DNN_MODEL_PATH}")

    # Save processed data
    df.to_csv(os.path.join(PROCESSED_DATA_DIR, "processed_data.csv"), index=False)
    pd.DataFrame({"feature": selected_features}).to_csv(
        os.path.join(PROCESSED_DATA_DIR, "selected_features.csv"), index=False
    )
    print(f"✓ Processed data saved")

    print("\n" + "="*60)
    print("Training pipeline completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
