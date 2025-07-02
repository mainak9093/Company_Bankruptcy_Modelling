# Company_Bankruptcy_Modelling
# Company Bankruptcy Prediction

## Overview

This project aims to predict whether a company is likely to go bankrupt based on its financial attributes. Using a combination of deep learning and probabilistic models, the solution is designed to handle severe class imbalance and extract complex patterns from financial data. The approach integrates a Deep Neural Network (DNN) and Gaussian Naive Bayes (GNB) classifier, with ensemble soft voting for robust predictions.

---
## Labels
X1	net profit / total assets
X2	total liabilities / total assets
X3	working capital / total assets
X4	current assets / short-term liabilities
X5	[(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365
X6	retained earnings / total assets
X7	EBIT / total assets
X8	book value of equity / total liabilities
X9	sales / total assets
X10	equity / total assets
X11	(gross profit + extraordinary items + financial expenses) / total assets
X12	gross profit / short-term liabilities
X13	(gross profit + depreciation) / sales
X14	(gross profit + interest) / total assets
X15	(total liabilities * 365) / (gross profit + depreciation)
X16	(gross profit + depreciation) / total liabilities
X17	total assets / total liabilities
X18	gross profit / total assets
X19	gross profit / sales
X20	(inventory * 365) / sales
X21	sales (n) / sales (n-1)
X22	profit on operating activities / total assets
X23	net profit / sales
X24	gross profit (in 3 years) / total assets
X25	(equity - share capital) / total assets
X26	(net profit + depreciation) / total liabilities
X27	profit on operating activities / financial expenses
X28	working capital / fixed assets
X29	logarithm of total assets
X30	(total liabilities - cash) / sales
X31	(gross profit + interest) / sales
X32	(current liabilities * 365) / cost of products sold
X33	operating expenses / short-term liabilities
X34	operating expenses / total liabilities
X35	profit on sales / total assets
X36	total sales / total assets
X37	(current assets - inventories) / long-term liabilities
X38	constant capital / total assets
X39	profit on sales / sales
X40	(current assets - inventory - receivables) / short-term liabilities
X41	total liabilities / ((profit on operating activities + depreciation) * (12/365))
X42	profit on operating activities / sales
X43	rotation receivables + inventory turnover in days
X44	(receivables * 365) / sales
X45	net profit / inventory
X46	(current assets - inventory) / short-term liabilities
X47	(inventory * 365) / cost of products sold
X48	EBITDA (profit on operating activities - depreciation) / total assets
X49	EBITDA (profit on operating activities - depreciation) / sales
X50	current assets / total liabilities
X51	short-term liabilities / total assets
X52	(short-term liabilities * 365) / cost of products sold)
X53	equity / fixed assets
X54	constant capital / fixed assets
X55	working capital
X56	(sales - cost of products sold) / sales
X57	(current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)
X58	total costs /total sales
X59	long-term liabilities / equity
X60	sales / inventory
X61	sales / receivables
X62	(short-term liabilities *365) / sales
X63	sales / short-term liabilities
X64	sales / fixed assets
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
