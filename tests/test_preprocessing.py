"""Unit tests for the preprocessing, feature-selection and evaluation modules.

    pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

# src/ is put on sys.path by tests/conftest.py

from config import TARGET_COLUMN, TRAIN_DATA_PATH
from data_preprocessing import (
    DataCleaner,
    find_constant_columns,
    find_error_columns,
    find_highly_correlated,
    load_data,
    split_features_target,
)
from feature_selection import get_selected_features_subset, select_features_anova
from model_evaluation import evaluate_model, find_optimal_threshold


def _toy_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = pd.Series(rng.binomial(1, 0.25, n), name=TARGET_COLUMN)
    X = pd.DataFrame(
        {
            "good_a": rng.normal(0.5, 0.1, n) + y * 0.4,
            "good_b": rng.normal(0.5, 0.1, n) - y * 0.3,
            "constant": np.ones(n),
            "noise": rng.normal(0.5, 0.1, n),
        }
    )
    X["copy_of_a"] = X["good_a"] * 1.001 + 1e-6  # near-perfectly correlated
    return X, y


# ------------------------------------------------------------------ loading --

def test_load_data_strips_column_whitespace():
    """Train.csv ships every feature name with a leading space.

    The original config.py listed the names WITHOUT that space, so every
    lookup silently missed and the pipeline crashed with a KeyError.
    """
    df = load_data(TRAIN_DATA_PATH)
    assert all(c == c.strip() for c in df.columns)
    assert TARGET_COLUMN in df.columns
    assert " ROA(C) before interest and depreciation before interest" not in df.columns
    assert "ROA(C) before interest and depreciation before interest" in df.columns


def test_split_features_target_rejects_missing_target():
    with pytest.raises(KeyError):
        split_features_target(pd.DataFrame({"a": [1, 2]}))


# ----------------------------------------------------------------- cleaning --

def test_find_constant_columns():
    X, _ = _toy_frame()
    assert find_constant_columns(X) == ["constant"]


def test_find_error_columns_splits_on_count():
    X = pd.DataFrame({
        "many_errors": [5.0] * 50 + [0.5] * 50,
        "few_errors": [5.0] * 3 + [0.5] * 97,
        "clean": [0.5] * 100,
    })
    high, low, counts = find_error_columns(X, threshold=2, max_errors=10)
    assert high == ["many_errors"]
    assert low == ["few_errors"]
    assert counts == {"many_errors": 50, "few_errors": 3}
    assert "clean" not in counts


def test_find_highly_correlated_keeps_stronger_target_signal():
    X, y = _toy_frame()
    dropped = find_highly_correlated(X[["good_a", "copy_of_a", "noise"]], y, threshold=0.90)
    assert len(dropped) == 1
    assert dropped[0] in {"good_a", "copy_of_a"}
    assert "noise" not in dropped


def test_cleaner_imputes_median_below_the_cap():
    """A repaired cell must never keep an out-of-range value.

    The original code took the median BEFORE masking, so an error value could
    survive imputation.
    """
    X = pd.DataFrame({
        "f": [0.1, 0.2, 0.3, 99.0, 0.4] * 20,
        "other": np.linspace(0, 1, 100),
    })
    y = pd.Series([0, 1] * 50)
    cleaner = DataCleaner(error_threshold=2, max_errors=50, verbose=False)
    out = cleaner.fit_transform(X, y)
    assert "f" in cleaner.low_error_columns_
    assert out["f"].max() <= 2
    assert cleaner.medians_["f"] <= 2


def test_cleaner_transform_is_deterministic_and_column_aligned():
    X, y = _toy_frame()
    cleaner = DataCleaner(verbose=False)
    fitted = cleaner.fit_transform(X, y)
    again = cleaner.transform(X.sample(frac=1.0, random_state=1).sort_index())
    assert list(fitted.columns) == list(again.columns) == cleaner.feature_names_
    assert "constant" not in fitted.columns
    pd.testing.assert_frame_equal(fitted, again)


def test_cleaner_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        DataCleaner(verbose=False).transform(pd.DataFrame({"a": [1.0]}))


def test_cleaner_rejects_frame_missing_required_columns():
    X, y = _toy_frame()
    cleaner = DataCleaner(verbose=False).fit(X, y)
    with pytest.raises(KeyError):
        cleaner.transform(X.drop(columns=["good_a"]))


def test_cleaner_learns_only_from_the_training_split():
    """Rules fitted on a subset must not change when unseen rows are transformed."""
    X, y = _toy_frame(n=400)
    train_idx, test_idx = np.arange(200), np.arange(200, 400)
    cleaner = DataCleaner(verbose=False).fit(X.iloc[train_idx], y.iloc[train_idx])
    medians_before = dict(cleaner.medians_)
    cleaner.transform(X.iloc[test_idx])
    assert cleaner.medians_ == medians_before


# --------------------------------------------------------- feature selection --

def test_select_features_anova_returns_requested_count():
    X, y = _toy_frame()
    X = X.drop(columns=["constant"])
    selected, scores = select_features_anova(X, y, top_n=2)
    assert len(selected) == 2
    assert set(selected).issubset(set(X.columns))
    assert len(scores) == X.shape[1]
    assert {"feature", "f_score", "p_value"} <= set(scores.columns)


def test_select_features_anova_clamps_top_n_to_width():
    X, y = _toy_frame()
    selected, _ = select_features_anova(X.drop(columns=["constant"]), y, top_n=999)
    assert len(selected) == X.shape[1] - 1


def test_select_features_anova_picks_the_informative_columns():
    X, y = _toy_frame()
    selected, _ = select_features_anova(X.drop(columns=["constant"]), y, top_n=2)
    assert "noise" not in selected


def test_get_selected_features_subset_fails_loudly():
    X, _ = _toy_frame()
    with pytest.raises(KeyError):
        get_selected_features_subset(X, ["not_a_column"])


# --------------------------------------------------------------- evaluation --

def test_find_optimal_threshold_can_reach_above_060():
    """SMOTE-trained models need thresholds well above 0.6.

    The original search stopped at 0.60 and could never find the optimum.
    """
    y_true = np.array([0] * 90 + [1] * 10)
    probs = np.concatenate([np.full(90, 0.80), np.full(10, 0.95)])
    thr, f1, _ = find_optimal_threshold(y_true, probs)
    assert thr > 0.60
    assert f1 == pytest.approx(1.0)


def test_evaluate_model_reports_accuracy_and_confusion_counts():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    m = evaluate_model(y_true, y_pred, np.array([0.1, 0.6, 0.4, 0.9]))
    assert m["tn"] == 1 and m["fp"] == 1 and m["fn"] == 1 and m["tp"] == 1
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert "roc_auc" in m and "pr_auc" in m


def test_evaluate_model_handles_a_degenerate_prediction():
    y_true = np.array([0] * 9 + [1])
    m = evaluate_model(y_true, np.zeros(10, dtype=int))
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1_score"] == 0.0
    assert m["accuracy"] == pytest.approx(0.9)


# -------------------------------------------------------------- integration --

def test_full_cleaning_on_the_real_dataset():
    """End-to-end smoke test on Train.csv: the exact path the original code crashed on."""
    df = load_data(TRAIN_DATA_PATH)
    X, y = split_features_target(df)
    assert X.shape == (5455, 95)
    assert int(y.sum()) == 154

    cleaner = DataCleaner(verbose=False)
    out = cleaner.fit_transform(X, y)

    assert out.isnull().sum().sum() == 0
    assert cleaner.constant_columns_ == ["Net Income Flag"]
    assert len(cleaner.high_error_columns_) == 8
    assert out.shape[1] == len(cleaner.feature_names_)
    assert out.shape[1] < X.shape[1]
    # low-error columns that survive correlation pruning must all be in range
    assert cleaner.repair_columns_
    for col in cleaner.repair_columns_:
        assert out[col].max() <= 2
    assert set(cleaner.repair_columns_) <= set(cleaner.low_error_columns_)
