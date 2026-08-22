"""Data loading and cleaning for bankruptcy prediction.

The cleaning rules are learned from the training split only (``fit``) and then
replayed on validation / test / production data (``transform``). Learning the
error columns, the medians and the correlation structure from the full dataset
- as the original code did - leaks test-set information into training.
"""

import numpy as np
import pandas as pd

from config import (
    ERROR_THRESHOLD,
    HIGH_CORRELATION_THRESHOLD,
    MAX_ERRORS_KEEP_COLUMN,
    TARGET_COLUMN,
)


def load_data(file_path):
    """Load a CSV and normalise its column names.

    Every feature column in Train.csv is stored with a leading space
    (``" ROA(C) before interest..."``). Stripping them here is what makes the
    names in ``config.py`` actually match the dataframe.
    """
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return df


def check_data_quality(df, verbose=True):
    """Report null cells and duplicate rows."""
    null_count = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    if verbose:
        print(f"  Null values: {null_count}")
        print(f"  Duplicate rows: {duplicate_count}")
    return null_count, duplicate_count


def find_constant_columns(df, exclude=()):
    """Columns with a single distinct value carry no information."""
    return [c for c in df.columns if c not in exclude and df[c].nunique(dropna=False) <= 1]


def find_error_columns(X, threshold=ERROR_THRESHOLD, max_errors=MAX_ERRORS_KEEP_COLUMN):
    """Split columns into unrecoverable (drop) and repairable (cap + impute).

    Returns ``(high_error, low_error, counts)`` where ``counts`` maps every
    affected column to the number of out-of-range rows.
    """
    counts = {}
    for col in X.columns:
        n_bad = int((X[col] > threshold).sum())
        if n_bad > 0:
            counts[col] = n_bad

    high_error = sorted([c for c, n in counts.items() if n > max_errors])
    low_error = sorted([c for c, n in counts.items() if 0 < n <= max_errors])
    return high_error, low_error, counts


def find_highly_correlated(X, y, threshold=HIGH_CORRELATION_THRESHOLD):
    """Greedily drop one of every feature pair correlated above ``threshold``.

    Of the two, the feature with the weaker absolute correlation to the target
    is dropped. ``y`` must come from the training split only.
    """
    features = list(X.columns)
    corr_matrix = X.corr().abs()
    target_corr = X.corrwith(y).abs().fillna(0.0)

    dropped = set()
    for i in range(len(features)):
        f1 = features[i]
        if f1 in dropped:
            continue
        for j in range(i + 1, len(features)):
            f2 = features[j]
            if f2 in dropped:
                continue
            pair_corr = corr_matrix.loc[f1, f2]
            if pd.notna(pair_corr) and pair_corr > threshold:
                if target_corr[f1] >= target_corr[f2]:
                    dropped.add(f2)
                else:
                    dropped.add(f1)
                    break  # f1 is gone; stop comparing against it
    return sorted(dropped)


class DataCleaner:
    """Learns cleaning rules on the training split and replays them elsewhere.

    Steps, in order:
      1. drop constant columns (e.g. ``Net Income Flag``, always 1)
      2. drop columns whose out-of-range count exceeds ``max_errors``
      3. cap remaining out-of-range cells and impute the training median
      4. drop one of each highly correlated feature pair
    """

    def __init__(
        self,
        error_threshold=ERROR_THRESHOLD,
        max_errors=MAX_ERRORS_KEEP_COLUMN,
        corr_threshold=HIGH_CORRELATION_THRESHOLD,
        verbose=True,
    ):
        self.error_threshold = error_threshold
        self.max_errors = max_errors
        self.corr_threshold = corr_threshold
        self.verbose = verbose

        self.constant_columns_ = []
        self.high_error_columns_ = []
        self.low_error_columns_ = []
        self.error_counts_ = {}
        self.medians_ = {}
        self.correlated_columns_ = []
        self.feature_names_ = []
        self.repair_columns_ = []

    def fit(self, X, y):
        X = X.copy()

        self.constant_columns_ = find_constant_columns(X)
        X = X.drop(columns=self.constant_columns_)

        high, low, counts = find_error_columns(X, self.error_threshold, self.max_errors)
        self.high_error_columns_ = high
        self.low_error_columns_ = low
        self.error_counts_ = counts
        X = X.drop(columns=self.high_error_columns_)

        # Medians are computed AFTER masking the bad cells, so an error value
        # can never drag the imputed value out of range.
        for col in self.low_error_columns_:
            masked = X[col].where(X[col] <= self.error_threshold, np.nan)
            median = masked.median(skipna=True)
            # A column that is entirely out of range would give NaN; fall back
            # to the cap so the output stays numeric.
            self.medians_[col] = float(median) if pd.notna(median) else float(self.error_threshold)
            X[col] = masked.fillna(self.medians_[col])

        self.correlated_columns_ = find_highly_correlated(X, y, self.corr_threshold)
        X = X.drop(columns=self.correlated_columns_)

        self.feature_names_ = list(X.columns)
        # Only the repaired columns that survive correlation pruning need to be
        # repaired again at transform time; requiring the rest would reject
        # otherwise-valid input frames.
        self.repair_columns_ = [c for c in self.low_error_columns_ if c in self.feature_names_]

        if self.verbose:
            print(f"  Constant columns dropped      : {len(self.constant_columns_)}")
            print(f"  High-error columns dropped    : {len(self.high_error_columns_)}")
            print(f"  Low-error columns repaired    : {len(self.low_error_columns_)}")
            print(f"  Correlated columns dropped    : {len(self.correlated_columns_)}")
            print(f"  Features remaining            : {len(self.feature_names_)}")
        return self

    def transform(self, X):
        if not self.feature_names_:
            raise RuntimeError("DataCleaner.transform called before fit")

        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise KeyError(f"Columns required by the fitted cleaner are missing: {missing}")

        X = X[self.feature_names_].copy()
        for col in self.repair_columns_:
            X[col] = X[col].where(X[col] <= self.error_threshold, np.nan)
            X[col] = X[col].fillna(self.medians_[col])

        return X

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


def split_features_target(df, target=TARGET_COLUMN):
    """Split a dataframe into (X, y)."""
    if target not in df.columns:
        raise KeyError(f"Target column {target!r} not found. Columns: {list(df.columns)[:5]}...")
    return df.drop(columns=[target]), df[target]
