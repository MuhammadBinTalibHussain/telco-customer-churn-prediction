# Telco Customer Churn Prediction

Predicting which telecom customers are likely to leave, using Logistic Regression, Decision Trees, Random Forest and XGBoost.

## Overview

Losing customers is expensive for a telecom company, so spotting at-risk customers early lets the company act before they leave. This project builds and compares four machine learning models on the Telco Customer Churn dataset to see which one identifies churners best.

## Dataset

- 7,043 customer records, 7,032 after cleaning (11 rows with blank `TotalCharges` were removed)
- 19 input features: demographics (gender, partner), account details (tenure, contract, payment method) and subscribed services (internet, phone, streaming)
- Target: `Churn` (Yes/No). About 26.6% of customers churned, so the classes are imbalanced.

## Approach

1. **Preprocessing:** converted `TotalCharges` to numeric, dropped missing rows, encoded `Churn` as 1/0, one-hot encoded categorical columns (first category dropped), made a stratified 80/20 train/test split, and standardized numeric features with the scaler fitted on training data only to avoid data leakage.
2. **Baseline:** Logistic Regression, with a coefficient chart showing which features raise or lower churn risk.
3. **Decision Trees:** compared an unconstrained tree with a pruned one (max depth 5, min 50 samples per leaf) to study overfitting, and visualized the pruned tree.
4. **Ensembles:** Random Forest (bagging, 100 trees) and XGBoost (boosting, log loss).
5. **Evaluation:** Accuracy, Precision, Recall, F1, a confusion matrix for the highest-Recall model, and ROC curves with AUC for all four models.

## Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.805 | 0.651 | 0.572 | 0.609 | 0.836 |
| Pruned Decision Tree | 0.775 | 0.598 | 0.473 | 0.528 | 0.819 |
| Random Forest | 0.788 | 0.623 | 0.513 | 0.563 | 0.816 |
| XGBoost | 0.771 | 0.576 | 0.527 | 0.550 | 0.811 |

## Key Findings

- **Higher churn risk:** fiber optic internet, electronic check payments, month-to-month contracts and streaming add-ons.
- **Lower churn risk:** long tenure, one or two year contracts, online security and tech support.
- **Overfitting:** the unconstrained tree scored 99.9% on training data but only 72.8% on test data. The pruned tree scored 79.3% and 77.5%, so it generalizes far better.
- **Best model:** Logistic Regression won on every metric with default settings. A simpler model can beat complex ones on this dataset.
- **Limitation:** even the best model catches only about 57% of churners at the default 0.5 threshold. Lowering the threshold or using class weights could improve Recall at the cost of more false alarms.

## How to Run

```bash
git clone https://github.com/<your-username>/telco-customer-churn-prediction.git
cd telco-customer-churn-prediction
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
jupyter notebook telco_churn_prediction.ipynb
```

Keep `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the same folder as the notebook.

## Repository Contents

- `telco_churn_prediction.ipynb`: full analysis with code, outputs and explanations
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: dataset
- `README.md`: project summary

## Tools

Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn
