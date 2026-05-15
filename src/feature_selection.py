"""Feature selection utilities using multiple methods."""

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif


def select_features_anova(X_data, y_data, top_n=30):
    """Select top features using ANOVA F-test."""
    selector = SelectKBest(score_func=f_classif, k=top_n)
    selector.fit(X_data, y_data)
    mask = selector.get_support()
    selected_features = X_data.columns[mask].tolist()

    # Get feature scores
    scores = selector.scores_
    feature_scores = pd.DataFrame(
        {"feature": X_data.columns, "score": scores}
    ).sort_values("score", ascending=False)

    return selected_features, feature_scores


def get_selected_features_subset(X_data, selected_features):
    """Extract selected features from dataset."""
    return X_data[selected_features]
