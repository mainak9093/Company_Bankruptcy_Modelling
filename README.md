# Company Bankruptcy Prediction

A machine learning project for predicting company bankruptcy using financial indicators and ensemble deep learning techniques.

## Overview

This project develops a hybrid ensemble model combining Deep Neural Networks (DNN) and Gaussian Naive Bayes (GNB) classifiers to predict bankruptcy risk based on 95 financial indicators from the company financial dataset. The model addresses the significant class imbalance (97.2% non-bankrupt vs 2.8% bankrupt) through data preprocessing, feature selection, and SMOTE oversampling.

### Key Features

- **Ensemble Approach**: Combines DNN and GaussianNB for robust predictions
- **Class Imbalance Handling**: SMOTE-based oversampling for minority class
- **Feature Selection**: ANOVA F-test for selecting top 30 discriminative features
- **Data Quality**: Robust preprocessing handling data errors and correlations
- **Production Ready**: Clean modular code structure with proper configuration management
- **High Performance**: 97.23% accuracy with 51.52% F1-score on imbalanced test set

## Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 97.23% |
| **Precision** | 48.57% |
| **Recall** | 54.84% |
| **F1-Score** | 51.52% |
| **ROC-AUC** | 0.9239 |
| **Optimal Threshold** | 0.50 |

### Dataset Details

- **Size**: 5,455 samples × 96 financial indicators
- **Target**: Binary classification (Bankrupt? 0/1)
- **Class Distribution**:
  - Non-bankrupt: 5,301 (97.2%)
  - Bankrupt: 154 (2.8%)
- **Data Quality**: Zero null values, zero duplicates after preprocessing

## Project Structure

```
.
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── config.py                # Configuration parameters
│   ├── data_preprocessing.py     # Data cleaning and preprocessing
│   ├── feature_selection.py      # Feature selection utilities
│   ├── model_training.py         # Model training functions
│   ├── model_evaluation.py       # Evaluation metrics
│   ├── train.py                  # Main training pipeline
│   └── predict.py                # Inference on new data
├── data/
│   ├── raw/                      # Original dataset
│   │   ├── Train.csv
│   │   └── sample.csv
│   └── processed/                # Processed data and features
├── models/
│   └── saved/                    # Trained model artifacts
│       ├── dnn_model.h5
│       ├── GaussianNB_model.pkl
│       └── scaler.pkl
├── docs/
│   └── reports/                  # Research papers and reports
├── notebooks/                    # Jupyter notebooks (legacy)
├── tests/                        # Unit tests
├── setup.py                      # Package setup configuration
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## Technical Approach

### 1. Data Preprocessing
- **Error Handling**: Removed 9 columns with >300 data errors (values >2)
- **Imputation**: Capped and filled 14 columns with low errors using median
- **Feature Dropping**: Removed 32 highly correlated features (>0.90 correlation)
- **Result**: Reduced from 96 to 63 features

### 2. Feature Selection
- **Method**: ANOVA F-test (SelectKBest)
- **Selected**: Top 30 features with highest discriminative power
- **Rationale**: Reduces dimensionality while retaining predictive signals

### 3. Class Imbalance Handling
- **Technique**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Effect**: Balanced training set (4,241 samples per class after oversampling)
- **Benefits**: Prevents model bias toward majority class

### 4. Model Architecture

#### Deep Neural Network (DNN)
```
Input Layer: 30 features
    ↓
Dense(256) + ReLU + BatchNorm + Dropout(0.5)
    ↓
Dense(128) + ReLU + BatchNorm + Dropout(0.5)
    ↓
Dense(64) + ReLU + BatchNorm + Dropout(0.4)
    ↓
Output: Dense(1) + Sigmoid
```
- **Optimizer**: Adam (lr=0.0005)
- **Loss**: Binary Crossentropy
- **Regularization**: Batch Normalization + Dropout

#### Gaussian Naive Bayes
- **Assumption**: Feature independence given class
- **Advantage**: Probabilistic and interpretable
- **Speed**: Fast inference

#### Ensemble Strategy
- **Method**: Soft voting (average probabilities)
- **DNN Probability** + **GNB Probability** / 2
- **Threshold**: 0.50 (optimized for F1-score)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/mainak9093/Company_Bankruptcy_Modelling.git
   cd Company_Bankruptcy_Modelling
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Optional: Install as package**
   ```bash
   pip install -e .
   ```

## Usage

### Training the Model

Run the complete training pipeline:

```bash
cd src
python train.py
```

This will:
1. Load and preprocess the training data
2. Select top 30 features using ANOVA
3. Apply SMOTE oversampling
4. Train DNN and GaussianNB models
5. Find optimal prediction threshold
6. Evaluate ensemble performance
7. Save models to `models/saved/`

**Expected output**:
```
============================================================
Company Bankruptcy Prediction - Training Pipeline
============================================================

[1/6] Loading data...
[2/6] Preprocessing data...
[3/6] Selecting features...
...
[6/6] Training models...
...
Ensemble (DNN + GaussianNB) Evaluation Results
Precision: 0.4857
Recall: 0.5484
F1-Score: 0.5152
ROC-AUC: 0.9239
```

### Making Predictions

Use the trained model to predict bankruptcy on new data:

```bash
cd src
python predict.py ../data/raw/sample.csv ../results/predictions.csv
```

**Input CSV format**: Should have the same financial indicator columns as training data

**Output CSV**: Contains:
- `DNN_Probability`: Probability from DNN
- `GNB_Probability`: Probability from GaussianNB
- `Ensemble_Probability`: Average probability
- `Prediction`: Binary prediction (0=Non-bankrupt, 1=Bankrupt)
- `Bankruptcy_Risk`: Risk level (Low/High)

**Example usage in Python**:

```python
import sys
sys.path.insert(0, 'src')
from predict import predict

results = predict('data/raw/sample.csv', 'results/predictions.csv')
print(results.head())
```

## Model Configuration

Edit `src/config.py` to modify:

```python
# Model hyperparameters
TOP_FEATURES = 30                    # Number of features to select
DNN_EPOCHS = 200                     # Training epochs
DNN_BATCH_SIZE = 64                  # Batch size
DNN_LEARNING_RATE = 0.0005          # Learning rate
ENSEMBLE_THRESHOLD = 0.50            # Decision threshold
```

## Experiments Explored

The project experimented with multiple approaches:

### 1. Random Forest + CatBoost
- **F1-Score**: 31.82%
- **Insights**: Good baseline, but lower recall

### 2. XGBoost + LightGBM + DNN Ensemble
- **F1-Score**: 41.46%
- **Insights**: Improved but still lower than final approach

### 3. ANOVA + DNN + GaussianNB (Final)
- **F1-Score**: 51.52%
- **Insights**: Best balanced performance, better recall for minority class

## Handling Class Imbalance

The 33:1 imbalance ratio (5301:154) is addressed through:

1. **SMOTE Oversampling**: Creates synthetic minority examples
2. **Class Weights**: Applied during DNN training (weight_minority ≈ 1.0 after SMOTE)
3. **Threshold Tuning**: Optimized threshold specifically for F1-score
4. **Balanced Metrics**: Reports precision, recall, and F1-score (not just accuracy)

## Files and Artifacts

### Data Files
- `data/raw/Train.csv`: Original training dataset (5455 × 96)
- `data/raw/sample.csv`: Example data for predictions
- `data/processed/processed_data.csv`: Cleaned dataset after preprocessing

### Model Artifacts
- `models/saved/dnn_model.h5`: Trained DNN (TensorFlow format)
- `models/saved/GaussianNB_model.pkl`: Trained GaussianNB (pickle format)
- `models/saved/scaler.pkl`: StandardScaler for feature normalization

### Documentation
- `docs/reports/Company-Bankruptcy-Prediction.pdf`: Project report
- `docs/reports/Report_IEEE.pdf`: IEEE-format technical paper

## Future Improvements

1. **Data**: Incorporate temporal features (time-series financial indicators)
2. **Features**: Add domain-specific financial ratios
3. **Models**: Try stacking or blending with gradient boosting models
4. **Deployment**: Create REST API for production inference
5. **Interpretability**: Add SHAP values for model explainability
6. **Monitoring**: Set up performance tracking for concept drift

## Testing

Run unit tests:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Submit a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@project{bankruptcy_prediction_2025,
  title={Company Bankruptcy Prediction using Ensemble Deep Learning},
  author={Mainak},
  year={2025},
  url={https://github.com/mainak9093/Company_Bankruptcy_Modelling}
}
```

## References

- Scikit-learn: [Imbalanced-learn SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- TensorFlow/Keras: [Deep Learning for Binary Classification](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)
- Financial Indicators: Standard financial ratios used in bankruptcy prediction literature

## Contact

For questions or feedback:
- GitHub: [@mainak9093](https://github.com/mainak9093)
- Email: (Add your email)

---

**Last Updated**: May 2025  
**Status**: Production Ready ✓
