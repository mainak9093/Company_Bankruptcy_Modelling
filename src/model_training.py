"""Model training utilities including DNN and GaussianNB models."""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from config import (
    DNN_EPOCHS,
    DNN_BATCH_SIZE,
    DNN_LEARNING_RATE,
    SMOTE_RANDOM_STATE,
)


def apply_smote(X_train, y_train):
    """Apply SMOTE to handle class imbalance."""
    smote = SMOTE(random_state=SMOTE_RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"SMOTE applied: {X_train.shape} -> {X_resampled.shape}")
    return X_resampled, y_resampled


def scale_features(X_train, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def calculate_class_weights(y_train):
    """Calculate class weights for imbalanced data."""
    majority_count = np.sum(y_train == 0)
    minority_count = np.sum(y_train == 1)
    weight_minority = majority_count / minority_count if minority_count > 0 else 1
    class_weights = {0: 1, 1: weight_minority}
    return class_weights


def create_dnn_model(input_dim):
    """Create Deep Neural Network model with batch normalization and dropout."""
    inputs = Input(shape=(input_dim,))

    x = Dense(256, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=DNN_LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    return model


def train_dnn(X_train, y_train, X_val, y_val, class_weights):
    """Train DNN model."""
    model = create_dnn_model(input_dim=X_train.shape[1])

    history = model.fit(
        X_train,
        y_train,
        epochs=DNN_EPOCHS,
        batch_size=DNN_BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        verbose=0,
    )

    return model, history


def train_gaussian_nb(X_train, y_train):
    """Train Gaussian Naive Bayes model."""
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def get_ensemble_predictions(dnn_model, gnb_model, X_test):
    """Get ensemble predictions by averaging probabilities."""
    dnn_probs = dnn_model.predict(X_test, verbose=0).flatten()
    gnb_probs = gnb_model.predict_proba(X_test)[:, 1]

    ensemble_probs = (dnn_probs + gnb_probs) / 2
    return ensemble_probs, dnn_probs, gnb_probs
