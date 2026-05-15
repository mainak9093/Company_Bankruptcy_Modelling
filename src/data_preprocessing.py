"""Data preprocessing utilities for bankruptcy prediction."""

import numpy as np
import pandas as pd
from config import (
    COLUMNS_HIGH_ERROR,
    COLUMNS_LOW_ERROR,
    ERROR_THRESHOLD,
    HIGH_CORRELATION_THRESHOLD,
)


def load_data(file_path):
    """Load data from CSV file."""
    return pd.read_csv(file_path)


def check_data_quality(df):
    """Check for null values and duplicates."""
    null_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()
    print(f"Null values: {null_count}")
    print(f"Duplicate rows: {duplicate_count}")
    return null_count, duplicate_count


def remove_unused_target_feature(df):
    """Remove Net Income Flag column as it provides no discriminative power."""
    if "Net Income Flag" in df.columns:
        df = df.drop(columns="Net Income Flag")
    return df


def handle_data_errors(df):
    """Handle data errors by dropping/capping problematic columns."""
    df = df.copy()

    df.drop(columns=COLUMNS_HIGH_ERROR, inplace=True, errors="ignore")

    for col in COLUMNS_LOW_ERROR:
        if col in df.columns:
            df[col] = df[col].where(df[col] <= ERROR_THRESHOLD, np.nan)

    df.fillna({col: df[col].median(skipna=True) for col in COLUMNS_LOW_ERROR}, inplace=True)

    return df


def drop_highly_correlated_features(df, threshold=HIGH_CORRELATION_THRESHOLD, target="Bankrupt?"):
    """Drop highly correlated features while keeping the one with higher target correlation."""
    df = df.copy()
    features = df.columns.drop("Bankrupt?")
    corr_matrix = df[features].corr().abs()
    target_corr = df[features].corrwith(df[target]).abs()

    dropped = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1 = features[i]
            f2 = features[j]
            if f1 in dropped or f2 in dropped:
                continue
            if corr_matrix.loc[f1, f2] > threshold:
                if target_corr[f1] >= target_corr[f2]:
                    dropped.append(f2)
                else:
                    dropped.append(f1)

    dropped = list(set(dropped))
    new_df = df.drop(columns=dropped)
    return new_df, dropped


def preprocess_data(df):
    """Complete preprocessing pipeline."""
    print("Starting data preprocessing...")

    df = remove_unused_target_feature(df)
    print("✓ Removed unused target feature")

    check_data_quality(df)

    df = handle_data_errors(df)
    print("✓ Handled data errors")

    df, dropped = drop_highly_correlated_features(df)
    print(f"✓ Dropped {len(dropped)} highly correlated features")
    print(f"Final dataset shape: {df.shape}")

    return df
