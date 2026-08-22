# Verified Results — numbers to quote in the report

Every figure below was produced by the code in `src/` on `data/raw/Train.csv`,
under a leak-free protocol, with seeds fixed. Regenerate with:

```bash
python src/train.py           # -> results/holdout_results.{csv,json}
python src/cross_validate.py  # -> results/cv_summary.{csv,json}, results/cv_per_fold.csv
python src/baselines.py       # -> results/baseline_results.{csv,json}
```

---

## 1. Dataset

| Property | Value |
|---|---|
| Rows | 5,455 |
| Raw columns | 96 (95 features + `Bankrupt?`) |
| Bankrupt (class 1) | 154 (2.82%) |
| Non-bankrupt (class 0) | 5,301 (97.18%) |
| Imbalance ratio | 34.4 : 1 |
| Null cells | 0 |
| Duplicate rows | 0 |

Source: Taiwan Economic Journal company bankruptcy data. Feature names in the
CSV carry a leading space (`" ROA(C) before interest..."`); `load_data()`
strips them.

> The class ratio is **34.4:1**, not the "25:1" or "33:1" stated in the old
> report and README.

## 2. Preprocessing (fitted on the training split only)

| Step | Columns removed | Remaining |
|---|---|---|
| Start | — | 95 |
| Constant columns (`Net Income Flag`, always 1) | 1 | 94 |
| Data-error columns (>300 rows above 2.0) | 8 | 86 |
| Highly correlated pairs (\|r\| > 0.90) | 21 | **65** |
| ANOVA F-test, top-30 | 35 | **30** |

15 further columns had 1–300 out-of-range rows; those cells were capped and
filled with the **training-split** median (columns repaired, not dropped).

The 8 dropped high-error columns:
`Total Asset Growth Rate`, `Research and development expense rate`,
`Cash Turnover Rate`, `Inventory Turnover Rate (times)`,
`Operating Expense Rate`, `Quick Asset Turnover Rate`,
`Fixed Assets Turnover Frequency`, `Current Asset Turnover Rate`.

> The old README said "removed 9 columns with >300 data errors" and "reduced
> from 96 to 63 features". The true counts are **8** and **65**.

## 3. Top 10 features by ANOVA F-score

| # | Feature | F-score | p-value |
|---|---|---|---|
| 1 | Net Income to Total Assets | 411.58 | 1.4e-86 |
| 2 | Total debt/Total net worth | 309.46 | 2.0e-66 |
| 3 | Borrowing dependency | 219.40 | 3.5e-48 |
| 4 | Retained Earnings to Total Assets | 209.42 | 3.9e-46 |
| 5 | Net Income to Stockholder's Equity | 180.07 | 4.5e-40 |
| 6 | Net worth/Assets | 177.94 | 1.3e-39 |
| 7 | Current Liabilities/Equity | 146.18 | 5.4e-33 |
| 8 | Persistent EPS in the Last Four Seasons | 140.57 | 8.1e-32 |
| 9 | Equity to Long-term Liability | 135.88 | 7.9e-31 |
| 10 | Current Liability to Assets | 118.15 | 4.3e-27 |

Full ranking: `data/processed/anova_feature_scores.csv`.

## 4. Headline result — 5-fold stratified cross-validation

**This is the number to put in the report.** A single 20% hold-out contains only
31 bankrupt companies, so one F1 value carries a very wide confidence interval.
Mean ± std over 5 folds, each fold re-running the entire pipeline:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| GaussianNB | 0.944 ± 0.007 | 0.265 ± 0.033 | 0.571 ± 0.141 | 0.360 ± 0.056 | 0.906 ± 0.030 | 0.269 ± 0.049 |
| DNN | 0.951 ± 0.022 | 0.326 ± 0.082 | 0.527 ± 0.100 | 0.391 ± 0.069 | 0.882 ± 0.059 | 0.342 ± 0.045 |
| **Ensemble (DNN+GNB)** | **0.960 ± 0.008** | **0.362 ± 0.081** | **0.461 ± 0.098** | **0.397 ± 0.058** | **0.908 ± 0.036** | **0.357 ± 0.047** |
| Ensemble @ fixed 0.50 | 0.934 ± 0.028 | 0.256 ± 0.079 | 0.590 ± 0.170 | 0.344 ± 0.083 | 0.908 ± 0.036 | 0.357 ± 0.047 |

Balanced accuracy: GNB 0.763 ± 0.067, DNN 0.745 ± 0.042, Ensemble 0.718 ± 0.047.
MCC: GNB 0.363, DNN 0.385, Ensemble 0.385.

**Statement for the report:** *the ensemble reaches ROC-AUC 0.908 ± 0.036 and
F1 0.397 ± 0.058 under 5-fold cross-validation on a dataset with a 34:1 class
imbalance.*

### Per-fold ensemble F1 (shows the spread)

| Fold | Threshold | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| 1 | 0.70 | 0.333 | 0.400 | 0.364 | 0.904 |
| 2 | 0.95 | 0.383 | 0.581 | 0.462 | 0.939 |
| 3 | 0.92 | 0.478 | 0.355 | 0.407 | 0.944 |
| 4 | 0.95 | 0.362 | 0.548 | 0.436 | 0.901 |
| 5 | 0.45 | 0.255 | 0.419 | 0.317 | 0.855 |

Fold F1 ranges 0.317 → 0.462. Any single hold-out number lands somewhere in
that band — which is exactly why the cross-validated mean is the honest figure.

## 5. Secondary result — single 80/20 hold-out

Same split geometry as the original report (1,091 test rows, 31 bankrupt), so
the confusion matrix is directly comparable to Figure 2 of `EE708_report.pdf`.

| Model | Thr | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| GaussianNB | 0.99 | 0.9505 | 0.3284 | 0.7097 | 0.4490 | 0.9509 | 0.3162 |
| DNN | 0.91 | 0.9569 | 0.3710 | 0.7419 | 0.4946 | 0.9315 | 0.4587 |
| **Ensemble** | 0.91 | **0.9588** | **0.3750** | **0.6774** | **0.4828** | **0.9595** | **0.4605** |
| Ensemble @ 0.50 | 0.50 | 0.9212 | 0.2430 | 0.8387 | 0.3768 | 0.9595 | 0.4605 |

Ensemble confusion matrix (rows = true 0/1, cols = predicted 0/1):

```
[[1025   35]
 [  10   21]]
```

TN 1025 · FP 35 · FN 10 · TP 21 — 21 of 31 bankruptcies caught, 35 false alarms.

## 6. Baseline comparison (identical protocol)

| Model | Thr | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.89 | 0.9514 | 0.3226 | 0.6452 | 0.4301 | 0.9435 | 0.3715 |
| GaussianNB | 0.99 | 0.9505 | 0.3284 | 0.7097 | 0.4490 | 0.9509 | 0.3162 |
| Decision Tree (depth 6) | 0.80 | 0.9294 | 0.2386 | 0.6774 | 0.3529 | 0.8944 | 0.2441 |
| Random Forest (300) | 0.45 | 0.9487 | 0.3188 | 0.7097 | 0.4400 | 0.9547 | 0.3885 |
| HistGradientBoosting | 0.08 | 0.9267 | 0.2525 | 0.8065 | 0.3846 | 0.9532 | 0.4510 |
| **DNN + GNB ensemble** | 0.91 | **0.9588** | **0.3750** | 0.6774 | **0.4828** | **0.9595** | **0.4605** |

The ensemble has the best F1, ROC-AUC and PR-AUC of the six, but the margin
over Random Forest and HistGradientBoosting is smaller than the fold-to-fold
variation — the honest claim is *competitive with, not decisively better than,
strong tree baselines*.

## 7. Findings worth writing up

1. **Threshold-free metrics are the trustworthy ones.** ROC-AUC (0.908) and
   PR-AUC (0.357) do not depend on the decision threshold; F1 does, and the
   tuned threshold varied from 0.45 to 0.95 across folds.
2. **SMOTE inflates the probability scale.** Training on a 50/50 resampled set
   and testing on a 2.8% positive set pushes the optimal threshold up to
   0.70–0.95. Searching only 0.30–0.60, as the original code did, cannot reach
   that optimum — which alone cost roughly 5 F1 points.
3. **Early stopping matters more than architecture.** Training the same network
   for a fixed 200 epochs with no callback drops test ROC-AUC from 0.932 to
   0.816. Measured directly:

   | Early-stopping rule | Stop epoch | Test F1 | Test ROC-AUC | Test PR-AUC |
   |---|---|---|---|---|
   | val PR-AUC, patience 20 | 26 | 0.4946 | 0.9315 | 0.4587 |
   | val PR-AUC, patience 40 | 46 | 0.4946 | 0.9315 | 0.4587 |
   | val ROC-AUC, patience 30 | 37 | 0.5169 | 0.9316 | 0.4648 |
   | val loss, patience 30 | 51 | 0.4839 | 0.8872 | 0.4580 |
   | **none (original, 200 epochs)** | 200 | **0.4507** | **0.8157** | **0.3711** |

4. **The ensemble helps ranking, not thresholded F1.** It has the best ROC-AUC
   and PR-AUC, but on the hold-out the DNN alone scores a slightly higher F1
   (0.4946 vs 0.4828). Soft voting with a poorly calibrated GaussianNB pulls
   scores toward the extremes.
5. **Accuracy is a useless headline here.** Predicting "never bankrupt" scores
   97.2% accuracy on this dataset. Any accuracy figure near 97% must be
   presented alongside precision, recall and F1.

## 8. Corrections to the previous report and README

| Claim in old README / `EE708_report.pdf` | Reality |
|---|---|
| Accuracy 97.23% | Not reproducible. The report's own confusion matrix gives 1059/1091 = **97.07%**, and this is accuracy at a tuned threshold on 31 positives. CV accuracy is **96.0%**. |
| F1 51.52%, Precision 48.57%, Recall 54.84% | Derived from the report's confusion matrix `[[1042,18],[14,17]]`, which no code in the repo produces. Verified F1 is **0.397 ± 0.058** (CV) / **0.483** (hold-out). |
| ROC-AUC 0.9239 | Appears in no notebook output. Verified ROC-AUC is **0.908 ± 0.036** (CV) / **0.960** (hold-out). |
| "Optimal threshold 0.50" | The report text says 0.45; the tuned value is **0.91** on the hold-out and 0.45–0.95 across folds. |
| Dataset 7,027 rows × 66 features, 6,756/271 | That is the **Polish** dataset used by the notebooks. The dataset in `data/raw/` is **Taiwan**: 5,455 × 96, 5,301/154. |
| Train 5,405 + 216, SMOTE to 5,405 each | Actual: fit split 3,491 rows (98 positive), SMOTE to 3,393 per class (6,786 total). |
| "Class ratio 25:1" | **34.4:1**. |
| "Removed 9 columns with >300 errors", "96 → 63 features" | **8** columns; **95 → 65** features. |
| RF+CatBoost F1 31.82%, XGB+LGBM+DNN F1 41.46% | The notebook that ran these reports **5.41%** and **~0%** — on the Polish dataset after a `dropna()` that deleted 88% of the bankrupt class. |

### Why the old numbers were irreproducible

* `notebooks/Code.ipynb` and `notebooks/EE708_project.ipynb` load
  `bankruptcy_raw.csv` (Polish, `id`/`class`/`Attr1..Attr64`) — a different
  dataset from `data/raw/Train.csv` (Taiwan, `Bankrupt?` + named ratios).
* Both notebooks call `pd.to_numeric(errors="coerce")` then `dropna()`, cutting
  7,027 rows to 3,194 and the bankrupt class from 271 to ~32. Their final cells
  print **F1 = 0.0000** and **F1 = 0.0252**.
* The report's text describes the Polish dataset while its confusion matrix
  (1,091 rows, 31 positives) comes from the Taiwan dataset — two different
  experiments spliced into one report.
* `src/train.py` crashed on line 1 of preprocessing and never ran at all
  (see `docs/BUGS.md`), so it produced no numbers either.
