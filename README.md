# Company Bankruptcy Prediction (EE708)

Predicting company bankruptcy from financial ratios with a DNN + Gaussian Naive
Bayes soft-voting ensemble, on a dataset with a 34:1 class imbalance.

> **Rebuilt May 2025 → August 2025.** The previous version of this repository
> could not run — `src/train.py` crashed in preprocessing — and the metrics in
> the README and the PDF report came from an experiment on a *different
> dataset*. Everything below is produced by the code in `src/`, is
> leak-free, and is reproducible from a fixed seed.
> See [`docs/BUGS.md`](docs/BUGS.md) for the 24 issues found and fixed,
> [`docs/RESULTS.md`](docs/RESULTS.md) for the verified numbers, and
> [`docs/reports/ERRATA.md`](docs/reports/ERRATA.md) for every wrong figure in
> the original report and slide deck next to its true value. The corrected
> report is [`docs/reports/EE708_report_corrected.pdf`](docs/reports/EE708_report_corrected.pdf).

---

## Results

**Headline (5-fold stratified cross-validation, mean ± std):**

| Metric | Ensemble (DNN + GNB) |
|---|---|
| ROC-AUC | **0.908 ± 0.036** |
| PR-AUC | **0.357 ± 0.047** |
| F1-Score | **0.397 ± 0.058** |
| Precision | 0.362 ± 0.081 |
| Recall | 0.461 ± 0.098 |
| Accuracy | 0.960 ± 0.008 |
| Balanced accuracy | 0.718 ± 0.047 |

**Single 80/20 hold-out** (1,091 test rows, 31 bankrupt; threshold 0.91 tuned on
a separate validation split):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| GaussianNB | 0.9505 | 0.3284 | 0.7097 | 0.4490 | 0.9509 | 0.3162 |
| DNN | 0.9569 | 0.3710 | 0.7419 | 0.4946 | 0.9315 | 0.4587 |
| **Ensemble** | **0.9588** | **0.3750** | 0.6774 | **0.4828** | **0.9595** | **0.4605** |

Confusion matrix: `[[1025, 35], [10, 21]]` — 21 of 31 bankruptcies caught at
35 false alarms.

> **On accuracy:** predicting "never bankrupt" scores 97.2% on this dataset.
> Accuracy is reported for completeness only; **PR-AUC and F1 are the
> meaningful metrics** at a 2.8% positive rate.

Full tables, baselines and per-fold spread: [`docs/RESULTS.md`](docs/RESULTS.md).

## Dataset

`data/raw/Train.csv` — Taiwan Economic Journal company financials.

| Property | Value |
|---|---|
| Rows | 5,455 |
| Columns | 96 (95 features + `Bankrupt?`) |
| Bankrupt | 154 (2.82%) |
| Non-bankrupt | 5,301 (97.18%) |
| Imbalance | 34.4 : 1 |
| Nulls / duplicates | 0 / 0 |

Note: every feature column name in the CSV has a **leading space**
(`" ROA(C) before interest..."`). `load_data()` strips them — this was the cause
of the original crash.

## Quick start

```bash
python -m venv venv
venv\Scripts\activate          # Windows;  source venv/bin/activate on Linux/macOS
pip install -r requirements.txt

python src/train.py            # train + evaluate + save artifacts  (~1 min CPU)
```

Run every script from the **project root**, not from `src/`.

## The scripts, and the order to run them

| # | Script | What it does | Needs | Writes |
|---|---|---|---|---|
| 1 | `src/train.py` | Trains on an 80/20 stratified hold-out, tunes the threshold on a validation split, evaluates once on test, saves all artifacts. | `data/raw/Train.csv` | `models/saved/*`, `data/processed/*`, `results/holdout_results.{csv,json}` |
| 2 | `src/cross_validate.py` | Re-runs the whole pipeline across 5 stratified folds. **These are the numbers to quote.** | `data/raw/Train.csv` | `results/cv_summary.{csv,json}`, `results/cv_per_fold.csv` |
| 3 | `src/baselines.py` | LogReg / GNB / Decision Tree / Random Forest / HistGradientBoosting under the identical protocol. | `data/raw/Train.csv` | `results/baseline_results.{csv,json}` |
| 4 | `src/predict.py` | Scores an unlabelled CSV with the saved artifacts. | step 1 | `results/predictions.csv` |
| 5 | `src/evaluate.py` | Scores a **labelled** CSV with the saved artifacts. | step 1 | stdout |
| 6 | `src/make_report_figures.py` | Regenerates every figure in the LaTeX report from the same pipeline that produces the metrics. | `data/raw/Train.csv` | `docs/reports/src/figures/` |

Steps 1–3 are independent of each other and can run in any order. Steps 4 and 5
require step 1 to have run first.

```bash
# Reproduce everything, in order
python src/train.py
python src/cross_validate.py
python src/baselines.py

# Then use the trained model
python src/predict.py data/raw/sample.csv results/predictions.csv
python src/evaluate.py path/to/labelled_test.csv

# Tests
pytest tests/ -v
```

Supporting modules (imported, not run directly): `config.py` (all settings and
paths), `data_preprocessing.py`, `feature_selection.py`, `model_training.py`,
`model_evaluation.py`, `pipeline.py` (the shared leak-free pipeline).

## Method

### Pipeline order

The ordering is the part the original code got wrong. Nothing below the split
may see the test set:

```
1. split off the test set              (stratified, 20%)
2. split a validation set off the rest (stratified, 20% — real class ratio, NO SMOTE)
3. fit the cleaner        on train only  → error columns, medians, correlations
4. fit ANOVA selection    on train only  → top 30 features
5. fit the StandardScaler on train only
6. SMOTE the SCALED training split only
7. train DNN (early-stop on val PR-AUC) + GaussianNB
8. tune the decision threshold on val
9. evaluate once on test
```

### Preprocessing

| Step | Removed | Remaining |
|---|---|---|
| Start | — | 95 |
| Constant columns (`Net Income Flag`) | 1 | 94 |
| Data-error columns (>300 rows above 2.0) | 8 | 86 |
| Correlated pairs (\|r\| > 0.90) | 21 | 65 |
| ANOVA top-30 | 35 | **30** |

15 more columns had 1–300 out-of-range cells; those cells are capped and filled
with the training-split median rather than dropping the column.

### Class imbalance

SMOTE on the scaled training split only: 3,491 rows → 6,786 (3,393 per class).
Validation and test keep the real 2.8% positive rate.

Class weights are **not** used — after SMOTE the classes are already balanced,
so inverse-frequency weights evaluate to `{0: 1, 1: 1.0}` and do nothing.

### Model

```
Input(30)
  → Dense(256) + ReLU + BatchNorm + Dropout(0.5)
  → Dense(128) + ReLU + BatchNorm + Dropout(0.5)
  → Dense(64)  + ReLU + BatchNorm + Dropout(0.4)
  → Dense(1)   + Sigmoid
```

Adam (lr 5e-4), binary cross-entropy, batch 64, up to 200 epochs with
**early stopping on validation PR-AUC** (patience 20, best weights restored).

Ensemble: soft voting, `(P_dnn + P_gnb) / 2`.

Early stopping is not cosmetic — training the same network for a fixed 200
epochs, as the original did, drops test ROC-AUC from 0.932 to **0.816**.

### Decision threshold

Tuned on the validation split over 0.01–0.99, never on the test set. Because the
models are trained on SMOTE-balanced data but tested at 2.8% prevalence, the
optimal threshold sits at **0.70–0.95**. The original code searched only
0.30–0.60 and could never reach it.

The tuned threshold varies from 0.45 to 0.95 across folds — a real instability
worth reporting, and the reason ROC-AUC and PR-AUC are the headline metrics.

## Configuration

Everything lives in [`src/config.py`](src/config.py):

```python
RANDOM_STATE   = 42     TOP_FEATURES = 30    TEST_SIZE = 0.2
CV_FOLDS       = 5      VAL_SIZE     = 0.2   USE_SMOTE = True
DNN_EPOCHS     = 200    DNN_BATCH_SIZE = 64  DNN_LEARNING_RATE = 0.0005
DNN_EARLY_STOPPING_PATIENCE = 20
ERROR_THRESHOLD = 2     MAX_ERRORS_KEEP_COLUMN = 300
HIGH_CORRELATION_THRESHOLD = 0.90
```

## Reproducibility

`set_global_seeds()` seeds Python, NumPy and TensorFlow. Two consecutive runs of
`train.py` and of `cross_validate.py` produce identical metrics to four decimal
places (verified). The original code seeded only `train_test_split`, so every
run gave different numbers.

Verified on Python 3.13.3, pandas 2.2.3, NumPy 2.2.6, scikit-learn 1.6.1,
imbalanced-learn 0.14.2, TensorFlow 2.21.0 (CPU) / Keras 3.15.1.

## Project structure

```
├── src/
│   ├── config.py               # all paths and hyperparameters
│   ├── data_preprocessing.py   # DataCleaner (fit on train, transform elsewhere)
│   ├── feature_selection.py    # ANOVA F-test selection
│   ├── model_training.py       # DNN, GaussianNB, SMOTE, scaling, seeding
│   ├── model_evaluation.py     # metrics + threshold tuning
│   ├── pipeline.py             # the shared leak-free pipeline
│   ├── train.py                # [run 1] hold-out training
│   ├── cross_validate.py       # [run 2] 5-fold CV — headline numbers
│   ├── baselines.py            # [run 3] baseline comparison
│   ├── predict.py              # [run 4] inference
│   └── evaluate.py             # [run 5] score a labelled CSV
├── data/
│   ├── raw/{Train.csv, sample.csv}
│   └── processed/              # selected_features.csv, anova_feature_scores.csv
├── models/
│   ├── saved/                  # current artifacts (regenerate with train.py)
│   └── legacy/                 # original artifacts — unusable, kept for provenance
├── results/                    # all generated metrics
├── docs/
│   ├── RESULTS.md              # verified numbers to quote
│   ├── BUGS.md                 # what was broken and how it was fixed
│   ├── ARCHITECTURE.md, QUICKSTART.md, PROJECT_SUMMARY.md
│   └── reports/
│       ├── EE708_report_corrected.pdf   # CURRENT report, numbers verified
│       ├── ERRATA.md                    # every wrong figure vs. its true value
│       ├── src/EE708_report.tex         # source of the corrected report
│       ├── src/figures/                 # generated by make_report_figures.py
│       ├── EE708_report.pdf             # original submission (superseded)
│       └── Company-Bankruptcy-Prediction.pdf  # original deck (superseded)
├── notebooks/                  # original Colab notebooks (different dataset — see BUGS.md)
└── tests/                      # pytest suite (18 tests)
```

## Comparing against the original code

The pre-rebuild `src/` lives in git history rather than being duplicated in
the tree. To read or diff it:

```bash
git show c229bb4:src/train.py          # the version that crashed
git diff c229bb4 HEAD -- src           # everything that changed
```

## Known limitations

1. **31 test positives.** Any single hold-out metric has a wide confidence
   interval; this is why the cross-validated mean ± std is the headline.
2. **Threshold instability.** F1-optimal thresholds range 0.45–0.95 across
   folds. Smoothed and bootstrap-median threshold rules were tested and gained
   ~0.014 F1 — inside one standard deviation, so the simple rule was kept.
3. **The ensemble's edge is narrow.** It has the best ROC-AUC and PR-AUC, but
   Random Forest and HistGradientBoosting are within fold-level noise on F1.
4. **No temporal split.** The dataset carries no year column, so the split is
   random rather than forward-in-time. A real deployment would need the latter.
5. **The `notebooks/` are legacy.** They run on a different (Polish) dataset and
   are kept only for provenance. `src/` is the project.

## License

MIT — see [LICENSE](LICENSE).
