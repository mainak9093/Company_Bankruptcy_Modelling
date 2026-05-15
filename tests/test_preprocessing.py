"""Unit tests for data preprocessing module."""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import (
    remove_unused_target_feature,
    handle_data_errors,
    drop_highly_correlated_features,
)


def test_remove_unused_target_feature():
    """Test removal of Net Income Flag column."""
    df = pd.DataFrame({
        'Bankrupt?': [0, 1],
        'Net Income Flag': [1, 1],
        'Feature1': [1.0, 2.0]
    })

    result = remove_unused_target_feature(df)
    assert 'Net Income Flag' not in result.columns
    assert 'Bankrupt?' in result.columns
    assert len(result.columns) == 2
    print("✓ test_remove_unused_target_feature passed")


def test_handle_data_errors():
    """Test data error handling."""
    df = pd.DataFrame({
        'Bankrupt?': [0, 1, 0],
        'Normal_Feature': [0.5, 0.6, 0.7],
        'Error_Feature': [1.0, 100.0, 0.8]  # Has value > 2
    })

    # Manually set Error_Feature for testing
    import config
    config.COLUMNS_LOW_ERROR = ['Error_Feature']

    result = handle_data_errors(df)
    assert result['Error_Feature'].max() <= 2
    print("✓ test_handle_data_errors passed")


def test_drop_highly_correlated_features():
    """Test removal of highly correlated features."""
    df = pd.DataFrame({
        'Bankrupt?': [0, 1, 0, 1, 0],
        'Feature1': [1.0, 2.0, 1.0, 2.0, 1.0],
        'Feature2': [1.0, 2.0, 1.0, 2.0, 1.0],  # Perfectly correlated with Feature1
        'Feature3': [2.0, 3.0, 2.0, 3.0, 2.0],  # Different
    })

    result, dropped = drop_highly_correlated_features(df, threshold=0.90)
    assert len(dropped) > 0
    assert 'Bankrupt?' in result.columns
    print("✓ test_drop_highly_correlated_features passed")


if __name__ == '__main__':
    test_remove_unused_target_feature()
    test_handle_data_errors()
    test_drop_highly_correlated_features()
    print("\nAll tests passed! ✓")
