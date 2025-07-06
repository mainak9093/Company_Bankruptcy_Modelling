# Company Bankruptcy Prediction

## Overview

This project aims to predict whether a company is likely to go bankrupt based on its financial attributes. Using a combination of deep learning and probabilistic models, the solution is designed to handle severe class imbalance and extract complex patterns from financial data. The approach integrates a Deep Neural Network (DNN) and Gaussian Naive Bayes (GNB) classifier, with ensemble soft voting for robust predictions.

---

## Dataset

- *Source:* bankruptcy_raw.csv.csv
- *Size:* 7,027 rows × 66 columns
- *Features:*  
  - Attr1 to Attr64: Numerical financial attributes for each company  
  - id: Unique identifier for each company  
  - class: Target variable (0 = Non-bankrupt, 1 = Bankrupt)

---

## Workflow

### 1. Exploratory Data Analysis (EDA)
- Visualize distributions, outliers, and missing values
- Analyze feature correlations and class imbalance

### 2. Data Preprocessing
- *Feature Selection:*  
  Selects the most relevant features using ANOVA F-scores
- *Imbalance Handling:*  
  Applies SMOTE to oversample the minority (bankrupt) class
- *Standardization:*  
  Scales all features to zero mean and unit variance with StandardScaler

### 3. Model Architecture

#### Deep Neural Network (DNN)
- *Input:* Selected features
- *Hidden Layers:*  
  - Layer 1: 256 neurons, ReLU, BatchNorm, Dropout (50%)  
  - Layer 2: 128 neurons, ReLU, BatchNorm, Dropout (50%)  
  - Layer 3: 64 neurons, ReLU, BatchNorm, Dropout (40%)
- *Output Layer:* 1 neuron, Sigmoid activation
- *Optimizer:* Adam (lr=0.0005)
- *Loss:* Binary Crossentropy

#### Gaussian Naive Bayes (GNB)
- Assumes feature independence and Gaussian likelihoods
- Computes posterior probabilities for bankruptcy

#### Ensemble
- Combines DNN and GNB predictions using soft voting
- Decision threshold tuned to maximize F1-score

### 4. Evaluation Metrics
- *Accuracy:* Overall correct predictions
- *Precision:* Proportion of predicted bankruptcies that are correct
- *Recall:* Proportion of actual bankruptcies detected
- *F1-Score:* Harmonic mean of precision and recall

---

## Results

- *Accuracy:* 97.23%
- *Precision:*  12.8%
- *Recall:* 66.67%
- *F1-Score:*  25.2%
- *Best Threshold:* 45%

Despite the extreme class imbalance (about 1:25), the hybrid model demonstrates strong performance, especially in identifying bankrupt companies with a balanced trade-off between precision and recall.

---

## Repository Structure

| File Name            | Description                                             |
|----------------------|--------------------------------------------------------|
| Code.ipynb | Complete implementation and experiments               |
| bankruptcy_raw.csv.csv          | Main dataset                                           |
| catboost_model.cbm |  catboost library used in training                                 |
| gnb_model.pkl | Pre-trained GNB model                                 |
| dnn_model.h5l       | Pre-trained DNN model                                  |
| lgbm_model.pkl        | Pre-trained LGBM model              |
| evaluation_results.json   | Notebook for loading models and making predictions     |
| Project_Report.pdf | Detailed project report                                |
| df.csv | Cleaned Data Frame                               |
| df_bankrupted.csv |  Data Frame for bankrupted                               |
| df_not_bankrupted.csv | Data Frame for not bankrupted                                |
|Labels.xlsx | Labels which are marked as attributes|
|Overview.pdf| Overview of this project|
|Report.pdf|Report for academic purposes.|

---

## How to Run

1. *Clone the repository:* 
2. *Install dependencies:*

3. *Run the evaluation notebook:*
- Open Evaluation.ipynb in Jupyter or Colab
- Load user/company data and obtain bankruptcy predictions

---

## Conclusion

The hybrid DNN + GNB ensemble effectively predicts company bankruptcy on highly imbalanced financial data. The model balances precision and recall, making it suitable for practical risk assessment. Future work may involve advanced ensemble techniques or additional domain-specific features for further improvement.

---
## Credits


Mainak Sarkar
CH Rishita
---
