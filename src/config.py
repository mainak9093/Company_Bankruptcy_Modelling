"""Configuration parameters for the bankruptcy prediction model.

All paths are absolute and derived from the project root, so scripts work
regardless of the directory they are launched from.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SAVED_MODELS_DIR = os.path.join(MODELS_DIR, "saved")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

TRAIN_DATA_PATH = os.path.join(RAW_DATA_DIR, "Train.csv")
SAMPLE_DATA_PATH = os.path.join(RAW_DATA_DIR, "sample.csv")

TARGET_COLUMN = "Bankrupt?"

# Artifacts written by train.py and consumed by predict.py / evaluate.py.
SCALER_PATH = os.path.join(SAVED_MODELS_DIR, "scaler.pkl")
GNB_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "gnb_model.pkl")
DNN_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "dnn_model.keras")
CLEANER_PATH = os.path.join(SAVED_MODELS_DIR, "cleaner.pkl")
SELECTED_FEATURES_PATH = os.path.join(PROCESSED_DATA_DIR, "selected_features.csv")
THRESHOLD_PATH = os.path.join(SAVED_MODELS_DIR, "threshold.json")

# ---------------------------------------------------------------- protocol --
RANDOM_STATE = 42
TEST_SIZE = 0.2          # held-out test fraction (1091 rows, 31 positives)
VAL_SIZE = 0.2           # validation fraction carved out of the training part
CV_FOLDS = 5             # folds for the cross-validated headline result
TOP_FEATURES = 30
SMOTE_RANDOM_STATE = 42
USE_SMOTE = True

# ------------------------------------------------------------ DNN settings --
DNN_EPOCHS = 200
DNN_BATCH_SIZE = 64
DNN_LEARNING_RATE = 0.0005
DNN_DROPOUT_1 = 0.5
DNN_DROPOUT_2 = 0.5
DNN_DROPOUT_3 = 0.4
DNN_UNITS = (256, 128, 64)
DNN_EARLY_STOPPING_PATIENCE = 20   # monitored on the real-distribution val set

# ---------------------------------------------------- preprocessing rules --
# The raw features are ratios that should live in [0, 1]. Values above
# ERROR_THRESHOLD are data-entry errors in the source dataset. A column with
# more than MAX_ERRORS_KEEP_COLUMN such rows is unrecoverable and is dropped;
# a column with fewer is repaired by capping the bad cells and imputing the
# training-set median.
ERROR_THRESHOLD = 2
MAX_ERRORS_KEEP_COLUMN = 300
HIGH_CORRELATION_THRESHOLD = 0.90

# Reference lists (what the rule above selects on the full Train.csv).
# These are documentation only -- the cleaner derives its own lists from the
# training split at fit time so that no test-set information leaks in.
COLUMNS_HIGH_ERROR_REFERENCE = [
    "Total Asset Growth Rate",
    "Research and development expense rate",
    "Cash Turnover Rate",
    "Inventory Turnover Rate (times)",
    "Operating Expense Rate",
    "Quick Asset Turnover Rate",
    "Fixed Assets Turnover Frequency",
    "Current Asset Turnover Rate",
]

COLUMNS_LOW_ERROR_REFERENCE = [
    "Interest-bearing debt interest rate",
    "Inventory/Current Liability",
    "Long-term Liability to Current Assets",
    "Cash/Current Liability",
    "Accounts Receivable Turnover",
    "Total assets to GNP price",
    "Average Collection Days",
    "Quick Ratio",
    "Allocation rate per person",
    "Total debt/Total net worth",
    "Revenue Per Share (Yuan \u00a5)",
    "Quick Assets/Current Liability",
    "Net Value Growth Rate",
    "Current Ratio",
    "Revenue per person",
    "Fixed Assets to Assets",
]

# Threshold used by predict.py when no tuned threshold.json is available.
ENSEMBLE_THRESHOLD = 0.50
