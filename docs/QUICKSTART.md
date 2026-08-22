# Quick Start

## 1. Install

```bash
python -m venv venv
venv\Scripts\activate          # Windows;  source venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

Python 3.9–3.13. TensorFlow CPU is sufficient; no GPU needed (training takes
under a minute).

## 2. Train

Run from the **project root** — not from inside `src/`.

```bash
python src/train.py
```

Takes ~1 minute on CPU. Writes:

* `models/saved/` — `cleaner.pkl`, `scaler.pkl`, `gnb_model.pkl`,
  `dnn_model.keras`, `threshold.json`
* `data/processed/` — `selected_features.csv`, `anova_feature_scores.csv`
* `results/holdout_results.{csv,json}`

Expected final block:

```
        model  threshold  accuracy  precision  recall  f1_score  roc_auc  pr_auc
          gnb     0.9900    0.9505     0.3284  0.7097    0.4490   0.9509  0.3162
          dnn     0.9100    0.9569     0.3710  0.7419    0.4946   0.9315  0.4587
     ensemble     0.9100    0.9588     0.3750  0.6774    0.4828   0.9595  0.4605
ensemble@0.50     0.5000    0.9212     0.2430  0.8387    0.3768   0.9595  0.4605
```

These numbers are deterministic — you should get them exactly.

## 3. Get the numbers to quote

```bash
python src/cross_validate.py
```

Takes ~4 minutes (5 folds). This is the result to put in a report:

```
ensemble:
  accuracy            0.9604 +/- 0.0078
  precision           0.3622 +/- 0.0810
  recall              0.4606 +/- 0.0983
  f1_score            0.3971 +/- 0.0576
  roc_auc             0.9084 +/- 0.0358
  pr_auc              0.3574 +/- 0.0471
```

## 4. Compare against baselines

```bash
python src/baselines.py
```

## 5. Predict on new data

```bash
python src/predict.py data/raw/sample.csv results/predictions.csv
```

The input CSV needs the same raw feature columns as `Train.csv`. A `Bankrupt?`
column is optional and ignored for prediction. Output columns:
`DNN_Probability`, `GNB_Probability`, `Ensemble_Probability`, `Prediction`,
`Bankruptcy_Risk`.

To score **labelled** data instead:

```bash
python src/evaluate.py path/to/labelled.csv
```

## 6. Tests

```bash
pytest tests/ -v      # 18 tests, ~2 s
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: config` | Running from inside `src/` | Run from the project root: `python src/train.py` |
| `ModuleNotFoundError: tensorflow` | Not installed | `pip install tensorflow-cpu` |
| `ModuleNotFoundError: imblearn` | Not installed | `pip install imbalanced-learn` |
| `FileNotFoundError: Missing artifacts` from `predict.py` | `train.py` has not been run | `python src/train.py` first |
| `KeyError: 'Bankrupt?'` | Input CSV has different column names | Column names are stripped on load; check the file really is the Taiwan dataset |
| Numbers differ from those above | Different library versions | Metrics are seeded and deterministic per environment; see README for verified versions |

## What to read next

* [`RESULTS.md`](RESULTS.md) — every verified number, with the corrections to
  the old report.
* [`BUGS.md`](BUGS.md) — the 24 bugs that were found and fixed.
* [`ARCHITECTURE.md`](ARCHITECTURE.md) — module-by-module design.
