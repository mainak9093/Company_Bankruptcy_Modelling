# Architecture Documentation

> **SUPERSEDED — August 2025.**
> The metrics and several structural details in this document are from the
> original release and are **not reproducible**. `src/train.py` crashed before
> producing any of them, and the figures trace back to an experiment on a
> different dataset. See `docs/RESULTS.md` for verified numbers and
> `docs/BUGS.md` for what was wrong. The design description below is still
> broadly accurate; the numbers are not.


## System Design Overview

The Company Bankruptcy Prediction system follows a modular, layered architecture designed for maintainability, testability, and production deployment.

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface Layer                    │
│         (Jupyter Notebooks, CLI, REST API)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                  Application Layer                          │
│     ├── train.py (Training Pipeline)                        │
│     └── predict.py (Inference Engine)                       │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                   Utility Layer                             │
│  ├── data_preprocessing.py                                  │
│  ├── feature_selection.py                                   │
│  ├── model_training.py                                      │
│  ├── model_evaluation.py                                    │
│  └── config.py                                              │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│               Infrastructure Layer                          │
│     (Data Files, Models, Scalers, Artifacts)               │
└─────────────────────────────────────────────────────────────┘
```

## Module Breakdown

### 1. Configuration Module (`config.py`)

**Purpose**: Centralized configuration management

**Key Components**:
- File paths (data, models)
- Hyperparameters (learning rate, epochs, batch size)
- Feature selection parameters
- Model thresholds

**Benefits**:
- Single source of truth for all settings
- Easy hyperparameter tuning
- Environment-specific configurations

### 2. Data Preprocessing Module (`data_preprocessing.py`)

**Purpose**: Data cleaning and preparation

**Functions**:
- `load_data()`: Load CSV files
- `check_data_quality()`: Validate data integrity
- `remove_unused_target_feature()`: Drop irrelevant columns
- `handle_data_errors()`: Fix data anomalies
- `drop_highly_correlated_features()`: Remove redundant features
- `preprocess_data()`: End-to-end pipeline

**Data Flow**:
```
Raw Data (5455×96)
    ↓
Remove Unused Features (5455×95)
    ↓
Handle Data Errors (5455×81)
    ↓
Drop Correlations (5455×63)
    ↓
Clean Data
```

### 3. Feature Selection Module (`feature_selection.py`)

**Purpose**: Dimensionality reduction while retaining predictive power

**Methods**:
- `select_features_anova()`: ANOVA F-test based selection
- `get_selected_features_subset()`: Extract selected columns

**Algorithm**:
1. Compute F-scores for each feature
2. Rank features by score
3. Select top K (default 30) features
4. Return selected feature names

**Result**: 30 most discriminative features

### 4. Model Training Module (`model_training.py`)

**Purpose**: Model training and ensemble creation

**Components**:

#### SMOTE Oversampling
```python
apply_smote(X_train, y_train)
# Input: (4364, 30) X, (4364,) y
# Output: (8482, 30) X_resampled, (8482,) y_resampled
# Effect: Balanced classes (4241 each)
```

#### Feature Scaling
```python
scale_features(X_train, X_test)
# StandardScaler: μ=0, σ=1
# Returns: scaled arrays and scaler object for inference
```

#### DNN Architecture
```
Input: 30 features
↓
Dense(256) → ReLU → BatchNorm → Dropout(0.5)
↓
Dense(128) → ReLU → BatchNorm → Dropout(0.5)
↓
Dense(64) → ReLU → BatchNorm → Dropout(0.4)
↓
Dense(1) → Sigmoid
↓
Output: Probability [0, 1]
```

#### Model Ensemble
```python
ensemble_prob = (dnn_prob + gnb_prob) / 2
prediction = 1 if ensemble_prob > threshold else 0
```

### 5. Model Evaluation Module (`model_evaluation.py`)

**Purpose**: Performance assessment and threshold optimization

**Metrics**:
- Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix
- Classification Report

**Threshold Optimization**:
```python
best_threshold, best_f1, results = find_optimal_threshold(
    y_test, y_pred_probs, start=0.30, end=0.60, step=0.01
)
# Searches for threshold that maximizes F1-score
```

### 6. Training Pipeline (`train.py`)

**Execution Flow**:

```
1. Load Data
   └─ data/raw/Train.csv

2. Preprocess
   ├─ Check quality
   ├─ Remove errors
   └─ Drop correlations

3. Feature Selection
   ├─ Compute ANOVA scores
   ├─ Select top 30
   └─ Create X_selected

4. Train-Test Split
   ├─ 80% train, 20% test
   └─ Stratified split (preserve class distribution)

5. Apply SMOTE
   ├─ Oversample minority
   └─ Balance classes

6. Scale Features
   ├─ Fit scaler on training
   └─ Transform test

7. Train Models
   ├─ DNN: 200 epochs, batch_size=64
   ├─ GaussianNB: probabilistic model
   └─ Save models

8. Generate Predictions
   └─ Soft voting ensemble

9. Optimize Threshold
   ├─ Search 0.30-0.60
   └─ Maximize F1-score

10. Evaluate
    ├─ Compute metrics
    ├─ Print results
    └─ Save artifacts
```

**Outputs**:
- `models/saved/dnn_model.h5`
- `models/saved/GaussianNB_model.pkl`
- `models/saved/scaler.pkl`
- `data/processed/processed_data.csv`
- `data/processed/selected_features.csv`

### 7. Inference Module (`predict.py`)

**Purpose**: Make predictions on new data

**Execution Flow**:

```
1. Load Artifacts
   ├─ Load saved DNN
   ├─ Load GaussianNB
   ├─ Load scaler
   └─ Load selected features

2. Prepare Data
   ├─ Load new data
   ├─ Select features
   └─ Scale features

3. Generate Predictions
   ├─ DNN forward pass
   ├─ GNB probabilities
   ├─ Average (soft voting)
   └─ Apply threshold

4. Output Results
   ├─ Probabilities from each model
   ├─ Ensemble prediction
   └─ Risk level
```

**Input/Output**:
```
Input: CSV with financial indicators
  ├─ Must have selected feature columns
  ├─ Any number of rows
  └─ Same format as training data

Output: CSV with predictions
  ├─ DNN_Probability
  ├─ GNB_Probability
  ├─ Ensemble_Probability
  ├─ Prediction (0/1)
  └─ Bankruptcy_Risk (Low/High)
```

## Data Flow Diagram

### Training Pipeline

```
Train.csv
    │
    ├──→ [Preprocessing]
    │     ├─ Clean errors
    │     ├─ Handle correlations
    │     └─ Output: Clean Data (5455×63)
    │
    ├──→ [Feature Selection]
    │     ├─ ANOVA F-test
    │     └─ Output: Top 30 features
    │
    ├──→ [Data Split]
    │     ├─ 80% train (4364)
    │     ├─ 20% test (1091)
    │     └─ Stratified split
    │
    ├──→ [SMOTE + Scaling]
    │     ├─ Oversample train
    │     ├─ StandardScaler
    │     └─ Output: Scaled data
    │
    ├──→ [Model Training]
    │     ├─ DNN: 200 epochs
    │     ├─ GaussianNB
    │     └─ Ensemble averaging
    │
    ├──→ [Threshold Optimization]
    │     └─ Grid search for best F1
    │
    └──→ [Evaluation & Save]
         ├─ Save models
         ├─ Save scaler
         └─ Save results
```

### Inference Pipeline

```
New Data
    │
    ├──→ [Load Artifacts]
    │     ├─ Load DNN model
    │     ├─ Load GaussianNB
    │     ├─ Load scaler
    │     └─ Load feature list
    │
    ├──→ [Prepare]
    │     ├─ Select features
    │     ├─ Scale
    │     └─ Format input
    │
    ├──→ [Predict]
    │     ├─ DNN inference
    │     ├─ GNB inference
    │     ├─ Average probs
    │     └─ Apply threshold
    │
    └──→ [Output]
         └─ Predictions.csv
```

## Class Imbalance Handling Architecture

```
Original Dataset
  Non-Bankrupt: 5301 (97.2%)
  Bankrupt: 154 (2.8%)
          ↓
    [Train Split]
  Non-Bankrupt: 4240
  Bankrupt: 123
          ↓
    [SMOTE]
  Non-Bankrupt: 4241
  Bankrupt: 4241 (synthetic)
          ↓
    [Training]
  Models learn balanced representation
          ↓
    [Test Split]
  Non-Bankrupt: 1060 (97.2%)
  Bankrupt: 31 (2.8%) - REAL data
          ↓
    [Evaluation]
  Realistic metrics on unbalanced test set
```

## Error Handling Strategy

### Data Preprocessing Errors

```
1. High Error Columns (>300 errors)
   └─ DROP completely
   
2. Low Error Columns (10-200 errors)
   └─ CAP values at threshold
   └─ FILL with median

3. Highly Correlated Features (r > 0.90)
   └─ KEEP feature with higher target correlation
   └─ DROP the other
```

### Model Errors

```
1. Class Imbalance
   └─ SMOTE oversampling
   └─ Class weights in DNN

2. Overfitting
   └─ Dropout layers
   └─ Batch normalization
   └─ Early stopping potential

3. Poor Threshold
   └─ Grid search optimization
   └─ Maximize F1-score
```

## Scalability Considerations

### Current Capacity
- **Data**: Up to 10,000 samples per file
- **Features**: Up to 100 features
- **Inference**: <1 second per sample

### Optimization Opportunities
1. **Batch Inference**: Process multiple samples efficiently
2. **Model Quantization**: Reduce model size for deployment
3. **Distributed Training**: For larger datasets
4. **Caching**: Cache preprocessing on repeated data

## Security Considerations

1. **Input Validation**: Verify data format and ranges
2. **Model Protection**: Serialized models should be validated
3. **Data Privacy**: Remove PII before sharing results
4. **Access Control**: Restrict model and data access

## Deployment Architecture

```
┌─────────────────────────────────────┐
│      REST API (Flask/FastAPI)       │
├─────────────────────────────────────┤
│  POST /predict (JSON)               │
│  GET /health (status check)         │
│  POST /batch_predict (CSV upload)   │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│  Prediction Service                 │
│  ├─ Load models (cached)            │
│  ├─ Validate input                  │
│  ├─ Run inference                   │
│  └─ Format output                   │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│  Model/Scaler Storage               │
│  (Production File System)            │
└─────────────────────────────────────┘
```

## Future Enhancements

1. **Model Versioning**: Track model performance over time
2. **A/B Testing**: Compare model variants in production
3. **Feature Store**: Centralized feature management
4. **Monitoring**: Track prediction drift and performance decay
5. **Explainability**: SHAP/LIME for prediction explanations

---

This architecture ensures:
✓ **Modularity**: Each component has single responsibility
✓ **Reusability**: Functions can be used independently
✓ **Testability**: Each module can be tested in isolation
✓ **Maintainability**: Clear separation of concerns
✓ **Scalability**: Easy to extend and optimize
