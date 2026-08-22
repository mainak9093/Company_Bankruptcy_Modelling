# Errata — reports in this directory

Both original PDFs in this directory contain metrics that no code in this
repository produces. This file records every wrong figure next to the verified
value.

| File | Status |
|---|---|
| `EE708_report.pdf` | Original submission. Superseded by `EE708_report_corrected.pdf`. |
| `EE708_report_corrected.pdf` | **Current.** Rebuilt from `src/EE708_report.tex`; every number regenerated from the code. |
| `Company-Bankruptcy-Prediction.pdf` | Original slide deck (Gamma export). No source available; not rebuilt. Its figures are corrected in the table below. |

Verified values: [`docs/RESULTS.md`](../RESULTS.md). Root cause of each defect:
[`docs/BUGS.md`](../BUGS.md).

---

## Why the original numbers were wrong

Three separate problems compounded:

1. **`src/train.py` never ran.** Every feature column in `Train.csv` carries a
   leading space (`" ROA(C) before interest..."`); the lists in `config.py` did
   not. Every column-drop silently matched nothing, and preprocessing then died
   with `KeyError: 'Revenue Per Share (Yuan ¥)'`. The pipeline produced no
   metric at all, on any run.

2. **The notebooks use a different dataset.** `notebooks/Code.ipynb` and
   `notebooks/EE708_project.ipynb` load `bankruptcy_raw.csv` — the *Polish*
   companies dataset (7,027 × 66, columns `id`, `class`, `Attr1..Attr64`).
   `data/raw/Train.csv` is the *Taiwan* dataset (5,455 × 96, named financial
   ratios). Both notebooks also call `pd.to_numeric(errors="coerce")` followed
   by `dropna()`, which cut 7,027 rows to 3,194 and the bankrupt class from 271
   to roughly 32. Their final cells print **F1 = 0.0000** and **F1 = 0.0252**.

3. **The report mixes the two experiments.** The prose of `EE708_report.pdf`
   describes the Polish dataset (7,027 rows, 66 features, 6,756/271 split,
   5,405 + 216 training rows) while its confusion matrix — 1,091 rows with 31
   positives — is a 20% split of the *Taiwan* dataset. The two halves of the
   report describe two different runs.

---

## `EE708_report.pdf` — corrections

| § | Original claim | Verified value |
|---|---|---|
| 2 | "7,027 rows … 66 financial features" | **5,455 rows, 96 columns** (95 features + target) |
| 2 | "6,756 non-bankrupt (97.2%) and 271 bankrupt (2.8%)" | **5,301 non-bankrupt (97.18%), 154 bankrupt (2.82%)** |
| 2 | "Features with over 800 erroneous entries were discarded, fewer than 200 median-imputed" | Threshold is **300**: 8 columns dropped, 15 repaired |
| 2 | "reduced the number of features from 63 to 50" | **95 → 65** after cleaning and correlation pruning |
| 3.B | "5,405 non-bankrupt and 216 bankrupt … balanced to 5,405 each" | Fitting split **3,393 + 98**, SMOTE to **3,393 per class** (6,786 rows) |
| 4 | "DNN alone yielded a maximum F1-score of 0.46" | DNN alone: **F1 = 0.4946** (hold-out), **0.391 ± 0.069** (CV) |
| 4 | "improving the F1-score to 0.51" | Ensemble: **F1 = 0.4828** (hold-out), **0.397 ± 0.058** (CV) |
| 4 | "The threshold (0.45) that maximized the F1-score" | **0.91**, tuned on validation. Across folds it ranges 0.45–0.95. |
| 4 | "trained for 200 epochs … 20% validation split" | Early stopping on validation PR-AUC (stops ≈ epoch 26). Fixed 200 epochs costs 0.12 ROC-AUC. |
| 5 | "reached 97.23% accuracy on the test set" | **95.88%** (hold-out), **96.04% ± 0.78%** (CV). Also inconsistent with the report's own Fig. 2: (1042+17)/1091 = **97.07%**. |
| 6 | "test accuracy of 97.23% and an F1-Score of 0.51" | **Accuracy 0.9588, F1 0.4828** (hold-out) |
| 6 | "class ratio of 25:1" | **34.4 : 1** |
| Fig. 2 | Confusion matrix `[[1042, 18], [14, 17]]` | `[[1025, 35], [10, 21]]` |
| Fig. 3 | Classification report: precision 0.49, recall 0.55, F1 0.52 | precision **0.3750**, recall **0.6774**, F1 **0.4828** |
| — | ROC-AUC not reported | **0.9595** (hold-out), **0.908 ± 0.036** (CV) |
| — | PR-AUC not reported | **0.4605** (hold-out), **0.357 ± 0.047** (CV) |

The corrected rebuild also adds what the original omitted: a cross-validated
result rather than a single split, a baseline comparison under the identical
protocol, PR-AUC (the appropriate summary at 2.8% prevalence), and an explicit
statement of the evaluation protocol.

---

## `Company-Bankruptcy-Prediction.pdf` — corrections

This deck describes the Polish dataset throughout. Its "Evaluation" slide
reports the output of `EE708_project.ipynb`, which is a near-random classifier.

| Slide | Original claim | Verified value |
|---|---|---|
| EDA | "7,027 rows and 66 columns (65 features, 1 target)" | **5,455 rows, 96 columns** |
| EDA | "Bankrupted 271 / Non-Bankrupted 6,756" | **154 / 5,301** |
| EDA | "16 features with over 800 errors were removed" | **8** columns, at a threshold of 300 |
| EDA | "19 features with fewer than 200 errors were corrected" | **15** columns repaired |
| EDA | "A total of 50 features were retained" / "reduced from 63 to 50" | **65** retained, then 30 after ANOVA |
| SMOTE | "5,397 samples of class 0 and 224 of class 1 … balanced to 5,397 each" | **3,393 and 98**, balanced to **3,393 each** |
| Standardisation | `fit_transform(X_train_sm)` — scaling applied *after* SMOTE | Order reversed: scale first, **then** SMOTE. SMOTE's k-NN search is meaningless on unscaled features. |
| Architecture | "Training: 200 epochs, batch size 64, 20% validation split" | Early stopping on validation PR-AUC; the validation split must not contain SMOTE samples |
| Evaluation | Precision **0.0128** | **0.3750** |
| Evaluation | Recall **0.6667** | **0.6774** |
| Evaluation | F1-Score **0.0252** | **0.4828** |
| Evaluation | Accuracy **0.9723** | **0.9588** |
| Evaluation | Best Threshold **0.4** | **0.91** |

Note that the deck's accuracy of 0.9723 is itself inconsistent with its own
source: the classification report printed by `EE708_project.ipynb` for that same
run shows an accuracy of **0.51**.

The "97.23%" figure that propagated into `README.md` and both résumés
originates from this slide, where it sits beside an F1 of 0.0252 — a model that
is barely better than chance.

---

## Reproducing the corrected numbers

```bash
python src/train.py                 # hold-out results
python src/cross_validate.py        # cross-validated headline
python src/baselines.py             # baseline comparison
python src/make_report_figures.py   # figures for the LaTeX report

cd docs/reports/src && pdflatex EE708_report.tex && pdflatex EE708_report.tex
```

All runs are seeded and deterministic.
