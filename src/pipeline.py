"""The leak-free end-to-end pipeline, shared by train.py and cross_validate.py.

Ordering matters, and the original code got several steps wrong. The correct
order is:

    1. split off the test set               (nothing below may see it)
    2. split a validation set off the train (real class distribution, no SMOTE)
    3. fit the cleaner on train only        (error columns, medians, correlations)
    4. fit ANOVA selection on train only
    5. fit the scaler on train only
    6. SMOTE the scaled training split only
    7. train DNN (early-stop on val) and GNB
    8. tune the decision threshold on val
    9. evaluate once on test
"""

import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    RANDOM_STATE,
    TOP_FEATURES,
    USE_SMOTE,
    VAL_SIZE,
)
from data_preprocessing import DataCleaner
from feature_selection import get_selected_features_subset, select_features_anova
from model_evaluation import evaluate_model, find_optimal_threshold
from model_training import (
    apply_smote,
    get_ensemble_predictions,
    scale_features,
    train_dnn,
    train_gaussian_nb,
)


def build_fold(
    X_train_full,
    y_train_full,
    X_test,
    y_test,
    top_features=TOP_FEATURES,
    use_smote=USE_SMOTE,
    random_state=RANDOM_STATE,
    val_size=VAL_SIZE,
    verbose=True,
):
    """Fit the whole pipeline on one train/test partition and evaluate it.

    Returns a dict holding the fitted objects, the tuned threshold and the
    metrics for the ensemble and both of its members.
    """
    # ---- 2. carve a real-distribution validation set out of the training part
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        stratify=y_train_full,
        random_state=random_state,
    )
    if verbose:
        print(f"  fit={X_fit.shape[0]} (pos {int(y_fit.sum())})  "
              f"val={X_val.shape[0]} (pos {int(y_val.sum())})  "
              f"test={X_test.shape[0]} (pos {int(np.sum(y_test))})")

    # ---- 3. clean, fitted on the fit split only
    cleaner = DataCleaner(verbose=verbose)
    X_fit_c = cleaner.fit_transform(X_fit, y_fit)
    X_val_c = cleaner.transform(X_val)
    X_test_c = cleaner.transform(X_test)

    # ---- 4. ANOVA feature selection, fitted on the fit split only
    selected_features, feature_scores = select_features_anova(X_fit_c, y_fit, top_n=top_features)
    X_fit_s = get_selected_features_subset(X_fit_c, selected_features)
    X_val_s = get_selected_features_subset(X_val_c, selected_features)
    X_test_s = get_selected_features_subset(X_test_c, selected_features)
    if verbose:
        print(f"  ANOVA selected {len(selected_features)} of {X_fit_c.shape[1]} features")

    # ---- 5. scale, fitted on the fit split only
    X_fit_sc, X_val_sc, X_test_sc, scaler = scale_features(X_fit_s, X_val_s, X_test_s)

    # ---- 6. SMOTE on the scaled training split only
    if use_smote:
        X_fit_bal, y_fit_bal = apply_smote(X_fit_sc, y_fit, verbose=verbose)
    else:
        X_fit_bal, y_fit_bal = X_fit_sc, np.asarray(y_fit)

    # ---- 7. train both models
    dnn_model, history, fit_info = train_dnn(
        X_fit_bal, y_fit_bal, X_val_sc, np.asarray(y_val), class_weights=None, seed=random_state
    )
    gnb_model = train_gaussian_nb(X_fit_bal, y_fit_bal)
    if verbose:
        print(f"  DNN stopped at epoch {fit_info['epochs_run']} "
              f"(best val PR-AUC at epoch {fit_info['best_epoch']})")

    # ---- 8. tune the threshold on validation, never on test
    val_ens, val_dnn, val_gnb = get_ensemble_predictions(dnn_model, gnb_model, X_val_sc)
    thresholds = {}
    for name, probs in (("ensemble", val_ens), ("dnn", val_dnn), ("gnb", val_gnb)):
        thr, val_f1, _ = find_optimal_threshold(np.asarray(y_val), probs)
        thresholds[name] = {"threshold": thr, "val_f1": val_f1}
    if verbose:
        print(f"  Threshold tuned on validation: {thresholds['ensemble']['threshold']:.2f} "
              f"(val F1 {thresholds['ensemble']['val_f1']:.4f})")

    # ---- 9. evaluate once on the untouched test set
    test_ens, test_dnn, test_gnb = get_ensemble_predictions(dnn_model, gnb_model, X_test_sc)
    y_test_arr = np.asarray(y_test)

    results = {}
    for name, probs in (("ensemble", test_ens), ("dnn", test_dnn), ("gnb", test_gnb)):
        thr = thresholds[name]["threshold"]
        y_pred = (probs >= thr).astype(int)
        results[name] = evaluate_model(y_test_arr, y_pred, probs)
        results[name]["threshold"] = thr

    # For reference: the ensemble at the untuned default of 0.50.
    results["ensemble@0.50"] = evaluate_model(
        y_test_arr, (test_ens >= 0.50).astype(int), test_ens
    )
    results["ensemble@0.50"]["threshold"] = 0.50

    return {
        "results": results,
        "thresholds": thresholds,
        "cleaner": cleaner,
        "scaler": scaler,
        "dnn_model": dnn_model,
        "gnb_model": gnb_model,
        "history": history,
        "fit_info": fit_info,
        "selected_features": selected_features,
        "feature_scores": feature_scores,
        # The cleaned fitting split, so figures drawn elsewhere use the exact
        # frame the model was fitted on rather than re-deriving their own.
        "X_fit_clean": X_fit_c,
        "probs": {"test_ensemble": test_ens, "test_dnn": test_dnn, "test_gnb": test_gnb},
        "splits": {
            "n_fit": int(X_fit.shape[0]), "n_fit_pos": int(y_fit.sum()),
            "n_val": int(X_val.shape[0]), "n_val_pos": int(y_val.sum()),
            "n_test": int(X_test.shape[0]), "n_test_pos": int(np.sum(y_test)),
            "n_after_smote": int(np.asarray(X_fit_bal).shape[0]),
        },
    }
