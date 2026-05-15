"""Configuration parameters for the bankruptcy prediction model."""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SAVED_MODELS_DIR = os.path.join(MODELS_DIR, "saved")

TRAIN_DATA_PATH = os.path.join(RAW_DATA_DIR, "Train.csv")
SAMPLE_DATA_PATH = os.path.join(RAW_DATA_DIR, "sample.csv")

SCALER_PATH = os.path.join(SAVED_MODELS_DIR, "scaler.pkl")
GNB_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "GaussianNB_model (1).pkl")
DNN_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "dnn_model.h5")

# Model hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
TOP_FEATURES = 30
SMOTE_RANDOM_STATE = 42

# DNN hyperparameters
DNN_EPOCHS = 200
DNN_BATCH_SIZE = 64
DNN_LEARNING_RATE = 0.0005
DNN_DROPOUT_1 = 0.5
DNN_DROPOUT_2 = 0.5
DNN_DROPOUT_3 = 0.4

# Ensemble parameters
ENSEMBLE_THRESHOLD = 0.50

# Columns to drop due to high data errors
COLUMNS_HIGH_ERROR = [
    "Total Asset Growth Rate",
    "Cash Turnover Rate",
    "Research and development expense rate",
    "Inventory Turnover Rate (times)",
    "Operating Expense Rate",
    "Quick Asset Turnover Rate",
    "Fixed Assets Turnover Frequency",
    "Current Asset Turnover Rate",
    "Interest-bearing debt interest rate",
]

# Columns to cap and fill with median
COLUMNS_LOW_ERROR = [
    "Revenue Per Share (Yuan ¥)",
    "Net Value Growth Rate",
    "Current Ratio",
    "Quick Ratio",
    "Total debt/Total net worth",
    "Accounts Receivable Turnover",
    "Average Collection Days",
    "Revenue per person",
    "Allocation rate per person",
    "Quick Assets/Current Liability",
    "Cash/Current Liability",
    "Inventory/Current Liability",
    "Long-term Liability to Current Assets",
    "Total assets to GNP price",
]

# Error thresholds
ERROR_THRESHOLD = 2
HIGH_CORRELATION_THRESHOLD = 0.90
