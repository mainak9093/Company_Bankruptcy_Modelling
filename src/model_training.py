"""Model construction and training: DNN, GaussianNB, and the soft-vote ensemble."""

import os
import random

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from config import (
    DNN_BATCH_SIZE,
    DNN_DROPOUT_1,
    DNN_DROPOUT_2,
    DNN_DROPOUT_3,
    DNN_EARLY_STOPPING_PATIENCE,
    DNN_EPOCHS,
    DNN_LEARNING_RATE,
    DNN_UNITS,
    RANDOM_STATE,
    SMOTE_RANDOM_STATE,
)


def set_global_seeds(seed=RANDOM_STATE):
    """Seed Python, NumPy and TensorFlow so a rerun reproduces the numbers.

    The original pipeline seeded neither NumPy nor TensorFlow, so the DNN - and
    therefore every reported metric - changed on every run.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def scale_features(X_train, *others):
    """Fit a StandardScaler on the training split and apply it to the rest."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    transformed = [scaler.transform(X) for X in others]
    return (X_train_scaled, *transformed, scaler)


def apply_smote(X_train, y_train, random_state=SMOTE_RANDOM_STATE, verbose=True):
    """Balance the training split with SMOTE.

    SMOTE must run on ALREADY SCALED data: it interpolates between k-nearest
    neighbours under a Euclidean metric, so on raw features the neighbour search
    is dominated by whichever column happens to have the largest units. The
    original pipeline ran SMOTE first and scaled afterwards.

    It must also run on the training split only - never on validation or test.
    """
    n_minority = int(np.sum(np.asarray(y_train) == 1))
    if n_minority < 2:
        if verbose:
            print("  SMOTE skipped: fewer than 2 minority samples")
        return X_train, y_train

    k_neighbors = min(5, n_minority - 1)
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    if verbose:
        print(f"  SMOTE: {np.asarray(X_train).shape} -> {np.asarray(X_resampled).shape}"
              f" (k_neighbors={k_neighbors})")
    return X_resampled, y_resampled


def calculate_class_weights(y_train):
    """Inverse-frequency class weights.

    Note: after SMOTE the classes are already balanced, so this returns
    ``{0: 1, 1: 1.0}`` and has no effect. It is only useful when SMOTE is
    disabled. The original code applied it on top of SMOTE, which is why the
    notebooks print ``Class Weights: {0: 1, 1: np.float64(1.0)}``.
    """
    y_train = np.asarray(y_train)
    majority_count = int(np.sum(y_train == 0))
    minority_count = int(np.sum(y_train == 1))
    weight_minority = majority_count / minority_count if minority_count > 0 else 1.0
    return {0: 1.0, 1: float(weight_minority)}


def create_dnn_model(input_dim, seed=RANDOM_STATE):
    """Build the 256-128-64 MLP with batch-norm and dropout."""
    initializer = tf.keras.initializers.GlorotUniform(seed=seed)

    inputs = Input(shape=(input_dim,))
    x = inputs
    for units, dropout in zip(DNN_UNITS, (DNN_DROPOUT_1, DNN_DROPOUT_2, DNN_DROPOUT_3)):
        x = Dense(units, activation="relu", kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout, seed=seed)(x)

    outputs = Dense(1, activation="sigmoid", kernel_initializer=initializer)(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=DNN_LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def train_dnn(X_train, y_train, X_val, y_val, class_weights=None, verbose=0, seed=RANDOM_STATE):
    """Train the DNN, early-stopping on validation PR-AUC.

    ``X_val`` must keep the real class distribution (no SMOTE): early stopping
    against a synthetically balanced set optimises for a distribution the model
    will never see at test time.
    """
    model = create_dnn_model(input_dim=X_train.shape[1], seed=seed)

    early_stopping = EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=DNN_EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=0,
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=DNN_EPOCHS,
        batch_size=DNN_BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[early_stopping],
        shuffle=True,
        verbose=verbose,
    )

    epochs_run = len(history.history["loss"])
    best_epoch = int(np.argmax(history.history["val_pr_auc"])) + 1
    return model, history, {"epochs_run": epochs_run, "best_epoch": best_epoch}


def train_gaussian_nb(X_train, y_train):
    """Train a Gaussian Naive Bayes classifier."""
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def get_ensemble_predictions(dnn_model, gnb_model, X):
    """Soft-vote: average the two probability outputs."""
    dnn_probs = dnn_model.predict(X, verbose=0).flatten()
    gnb_probs = gnb_model.predict_proba(X)[:, 1]
    ensemble_probs = (dnn_probs + gnb_probs) / 2.0
    return ensemble_probs, dnn_probs, gnb_probs
