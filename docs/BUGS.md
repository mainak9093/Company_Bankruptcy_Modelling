# Bug report — what was broken and what was changed

Ordered by severity. Every entry names the original file and line, what went
wrong, and where the fix now lives.

---

## Blockers — the pipeline could not run

### B1. Column-name mismatch crashed preprocessing immediately
**`src/config.py:37–65` + `src/data_preprocessing.py:38–44`**

Every feature column in `Train.csv` is stored with a **leading space**:

```
Bankrupt?, ROA(C) before interest and depreciation before interest, ...
           ^
```

`COLUMNS_HIGH_ERROR` and `COLUMNS_LOW_ERROR` listed the names *without* it, so:

* `df.drop(columns=COLUMNS_HIGH_ERROR, errors="ignore")` matched nothing — the
  9 "dropped" columns were all still in the data;
* `for col in COLUMNS_LOW_ERROR: if col in df.columns:` was never true — no
  capping happened;
* `remove_unused_target_feature` never found `"Net Income Flag"`;
* then line 44 built `{col: df[col].median() for col in COLUMNS_LOW_ERROR}`
  **without** the `if col in df.columns` guard and raised:

```
KeyError: 'Revenue Per Share (Yuan ¥)'
```

`python src/train.py` therefore died in step 2 of 6, every time. **None of the
numbers in the README or the PDF report came from this code.**

*Fixed:* `load_data()` now calls `df.columns.str.strip()`, and the cleaner
derives its column lists from the data instead of a hand-typed list.

### B2. `predict.py` could never find its feature list
**`src/predict.py:36–37`**

```python
processed_dir = os.path.dirname(SCALER_PATH).replace("saved", "")
features_path = os.path.join(processed_dir, "processed", "selected_features.csv")
```

`os.path.dirname(SCALER_PATH)` is `models/saved`; stripping `"saved"` gives
`models/`, so it looked in **`models/processed/selected_features.csv`** while
`train.py` writes to **`data/processed/selected_features.csv`**. Guaranteed
`FileNotFoundError` on every invocation.

*Fixed:* uses `SELECTED_FEATURES_PATH` from config directly.

### B3. The saved models do not match any code in the repository
**`models/saved/`**

`scaler.pkl` and `GaussianNB_model (1).pkl` expect the **Taiwan** feature names
and were fitted with `class_count_ = [4241, 4241]` — consistent with
`Train.csv` (154 × 0.8 → SMOTE). But both notebooks in `notebooks/` train on the
**Polish** dataset (`bankruptcy_raw.csv`, `Attr1..Attr64`). The artifacts came
from a Colab session that is not in the repository and cannot be regenerated.

Scored against all of `Train.csv` — data it was partly fitted on — the saved DNN
reports F1 = 0.87. That is memorisation, not performance.

*Fixed:* moved to `models/legacy/` with a provenance note;
`python src/train.py` regenerates a matched, reproducible set.

---

## Methodology — the pipeline ran but the numbers were not valid

### B4. Feature selection saw the test set
**`src/train.py:50–62`** — ANOVA ran on the **full** dataset, and only
afterwards was the data split. Every test row helped choose which 30 features
the model was allowed to use.

*Fixed:* `pipeline.py` splits first; `select_features_anova` is fitted on the
training split only.

### B5. Correlation pruning and median imputation saw the test set
**`src/data_preprocessing.py:44, 49–71`** — `df[col].median()` and
`df.corrwith(df[target])` were both computed over all 5,455 rows, including the
test rows, and `corrwith(target)` leaks the **labels** as well.

*Fixed:* `DataCleaner.fit()` learns medians and correlations from the training
split; `.transform()` replays them.

### B6. The decision threshold was tuned on the test set and then reported on it
**`src/train.py:95–104`**

```python
best_threshold, best_f1, _ = find_optimal_threshold(y_test, ensemble_probs)
y_pred = (ensemble_probs > best_threshold).astype(int)
metrics = evaluate_model(y_test, y_pred, ensemble_probs)
```

The threshold was chosen to maximise F1 *on the test set*, then the resulting F1
was reported as test performance. That number is optimistically biased by
construction.

*Fixed:* the threshold is tuned on a separate validation split that the models
never train on; the test set is touched exactly once.

### B7. SMOTE ran before scaling
**`src/train.py:67–68`** — `apply_smote(X_train, y_train)` then
`scale_features(X_train_sm, X_test)`.

SMOTE interpolates between k-nearest neighbours under a Euclidean metric. On raw
features that span from `0.0–1.0` (`Debt ratio %`) to `10^9`
(`Operating Expense Rate`), the neighbour search is decided almost entirely by
the large-magnitude columns, so the synthetic minority samples are meaningless.

*Fixed:* scale first, then SMOTE on the scaled training split.

### B8. The validation set was made of synthetic SMOTE samples
**`src/train.py:75–77`** — the validation split was carved out **after** SMOTE,
so it was 50/50 balanced and contained synthetic rows interpolated from
training rows. Validation metrics measured a distribution that does not exist at
test time, and a synthetic row's parents sat in the training set.

*Fixed:* validation is split off **before** any resampling and keeps the real
2.8% positive rate. SMOTE touches the training split only.

### B9. The threshold search could not reach the optimum
**`src/model_evaluation.py:14`** — `find_optimal_threshold(..., start=0.30, end=0.60)`.

Because the models are trained on a SMOTE-balanced set, their probabilities are
inflated and the F1-optimal threshold on real-prevalence data lands at
**0.70–0.95**. The search window could never get there. It also silently
returned 0.5 when every threshold scored 0.

*Fixed:* search spans 0.01–0.99; the returned best-F1 is the real maximum.

### B10. Nothing was seeded, so no result was reproducible
**`src/train.py`** seeded only `train_test_split`. NumPy, Python `random` and
TensorFlow were left unseeded, so every run produced different weights, a
different threshold and different metrics. Quoting a metric to four decimals
from such a run is not meaningful.

*Fixed:* `set_global_seeds()` seeds Python, NumPy and TensorFlow. Two
consecutive `python src/train.py` runs now produce byte-identical metrics
(verified).

### B11. 200 epochs with no early stopping — the model overfits
**`src/model_training.py:76–84`** — fixed 200 epochs, no callbacks, and the
`validation_data` it was passed was the contaminated set from B8.

Measured on the fixed pipeline: fixed-200-epoch training gives test ROC-AUC
**0.816**; early stopping on validation PR-AUC gives **0.932**.

*Fixed:* `EarlyStopping(monitor="val_pr_auc", patience=20,
restore_best_weights=True)` against the real-distribution validation set.

### B12. Class weights on top of SMOTE are a no-op
**`src/train.py:71` / `model_training.py:35–41`** — `calculate_class_weights`
was called on the **already balanced** `y_train_sm`, so it always returned
`{0: 1, 1: 1.0}`. The notebooks print exactly that:
`Class Weights: {0: 1, 1: np.float64(1.0)}`. The README nevertheless lists
"Class Weights" as one of three techniques used to handle imbalance.

*Fixed:* the function is kept (it is useful when SMOTE is disabled) with a
docstring stating it has no effect after SMOTE; `pipeline.py` passes
`class_weights=None` and the README no longer claims it.

---

## Correctness and robustness

### B13. Median imputation could leave out-of-range values
**`src/data_preprocessing.py:44`** — the median was taken **after** the bad
cells became `NaN` in the loop above, but the dict comprehension re-read
`df[col]`, which at that point still contained the un-masked column for any
column that failed the `in df.columns` guard. If a column were majority-error,
the imputed median would itself be out of range.

*Fixed:* the median is computed from the explicitly masked series and falls back
to the cap; a unit test asserts `out[col].max() <= 2`.

### B14. `drop_highly_correlated_features` ignored its own `target` parameter
**`src/data_preprocessing.py:49–52`** — signature took `target="Bankrupt?"` but
line 52 hard-coded `df.columns.drop("Bankrupt?")`, so passing any other target
name raised `KeyError`.

*Fixed:* the target is a parameter throughout; `find_highly_correlated` now
takes `X` and `y` separately and cannot mix them up.

### B15. The correlation loop kept comparing against already-dropped features
**`src/data_preprocessing.py:57–67`** — when `f1` was added to `dropped`, the
inner loop carried on comparing `f2` against it for the rest of the row.

*Fixed:* `break` out of the inner loop as soon as `f1` is dropped.

### B16. `predict.py` skipped preprocessing entirely
**`src/predict.py:74`** — it sub-selected the 30 columns and scaled them, but
never applied the error capping and median imputation used at training time. A
production row with an out-of-range value was scaled as-is, far outside the
range the model was trained on.

*Fixed:* `prepare_features()` applies the pickled `DataCleaner`, so training and
inference share one transform chain.

### B17. `evaluate_model` never returned accuracy
**`src/model_evaluation.py:33–53`** — accuracy was absent from the metric dict
and from `print_metrics`, yet "Accuracy: 97.23%" was the headline of both the
README and the report.

*Fixed:* accuracy, balanced accuracy, MCC, PR-AUC and the four confusion-matrix
cells are all returned and printed. PR-AUC is the appropriate summary at a 2.8%
positive rate.

### B18. `predict.py` crashed when writing to `results/`
**`src/predict.py:104`** — the README documents
`python predict.py ../data/raw/sample.csv ../results/predictions.csv`, but
nothing created `results/`, so `to_csv` raised `FileNotFoundError`.

*Fixed:* the output directory is created if missing.

### B19. `.h5` model format is legacy under Keras 3
**`src/train.py:118`** — `model.save("dnn_model.h5")` emits a deprecation
warning under Keras 3 (the environment here is Keras 3.15).

*Fixed:* saves `dnn_model.keras`, the native format.

### B20. Model path contained a browser download artifact
**`src/config.py:16`** — `"GaussianNB_model (1).pkl"`. The `(1)` is the
duplicate-download marker a browser appends; a space in a path is a hazard in
shell commands.

*Fixed:* `gnb_model.pkl`.

### B21. Tests mutated global config and asserted almost nothing
**`tests/test_preprocessing.py:41–46`** — `test_handle_data_errors` did
`import config; config.COLUMNS_LOW_ERROR = ['Error_Feature']` after
`data_preprocessing` had already imported the value, so the reassignment had no
effect on the code under test; the test then passed vacuously. Tests also
printed instead of asserting and were not runnable under `pytest` as documented.

*Fixed:* 18 real tests, no global mutation, including a regression test for the
leading-space bug (B1) and for the 0.30–0.60 threshold window (B9).

---

## Documentation

### B22. Reported metrics match no code in the repository
See `docs/RESULTS.md` §8. The README's F1 (51.52%), precision (48.57%) and
recall (54.84%) are back-derived from the confusion matrix in
`EE708_report.pdf`; the accuracy quoted (97.23%) does not even match that matrix
(1059/1091 = 97.07%); and the ROC-AUC (0.9239) appears in no notebook output.
The notebooks that *are* in the repo print F1 = 0.0000 and F1 = 0.0252.

### B23. The report describes a different dataset from the one shipped
`EE708_report.pdf` §2 describes 7,027 rows and 66 features with 6,756/271 class
counts — the Polish dataset. `data/raw/Train.csv` is the Taiwan dataset:
5,455 × 96 with 5,301/154. The report's own confusion matrix (1,091 rows, 31
positives) is from the Taiwan data. The two halves of the report describe two
different experiments.

### B24. Counts in the README do not match the data
"9 columns with >300 errors" → 8. "Reduced from 96 to 63 features" → 95 to 65.
"33:1 imbalance" and the report's "25:1" → 34.4:1. "4,241 samples per class
after oversampling" → 3,393 per class under the corrected split.

---

## Summary

| Category | Count |
|---|---|
| Blockers (code could not run / artifacts unusable) | 3 |
| Invalid methodology (leakage, wrong ordering) | 9 |
| Correctness & robustness | 9 |
| Documentation inaccuracies | 3 |
| **Total** | **24** |
