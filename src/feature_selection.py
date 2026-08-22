"""Feature selection using the ANOVA F-test.

``SelectKBest`` must be fitted on the training split only. Fitting it on the
full dataset - as the original ``train.py`` did - lets the test rows influence
which features the model is allowed to see.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif


def select_features_anova(X_data, y_data, top_n=30):
    """Rank features by ANOVA F-score and return the top ``top_n``.

    Returns ``(selected_features, feature_scores)``. Features whose score is
    undefined (constant columns) are pushed to the bottom rather than kept by
    accident.
    """
    top_n = min(top_n, X_data.shape[1])

    selector = SelectKBest(score_func=f_classif, k=top_n)
    selector.fit(X_data, y_data)

    scores = np.nan_to_num(selector.scores_, nan=-np.inf)
    p_values = selector.pvalues_

    feature_scores = pd.DataFrame(
        {"feature": X_data.columns, "f_score": selector.scores_, "p_value": p_values}
    ).sort_values("f_score", ascending=False, na_position="last")

    order = np.argsort(-scores, kind="stable")[:top_n]
    selected_features = [X_data.columns[i] for i in sorted(order)]

    return selected_features, feature_scores.reset_index(drop=True)


def get_selected_features_subset(X_data, selected_features):
    """Extract the selected columns, failing loudly if any are missing."""
    missing = [c for c in selected_features if c not in X_data.columns]
    if missing:
        raise KeyError(f"Selected features missing from data: {missing}")
    return X_data[selected_features]
