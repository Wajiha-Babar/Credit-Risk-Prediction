---
title: Credit Risk Prediction Dashboard
emoji: 💳
colorFrom: red
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
---

# Credit Risk Prediction

## Objective
The objective of this project is to predict whether a loan applicant is likely to default on a loan.

## Dataset
This project uses the Kaggle Give Me Some Credit dataset.

The target column is `SeriousDlqin2yrs`.

## Approach
1. Loaded the dataset.
2. Removed unnecessary columns.
3. Handled missing values using median imputation.
4. Treated outliers using percentile-based capping.
5. Performed exploratory data analysis using graphs.
6. Trained Logistic Regression and Decision Tree models.
7. Evaluated models using Accuracy, Precision, Recall, F1 Score, ROC-AUC, MAE, RMSE and Confusion Matrix.
8. Saved the best model for dashboard deployment.

## Machine Learning Models
- Logistic Regression
- Decision Tree Classifier

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- MAE
- RMSE
- Confusion Matrix

## Dashboard
The dashboard is built with Streamlit. It allows users to enter applicant details and predict credit default risk.

## Tools and Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Plotly
- Joblib

## Conclusion
The project successfully predicts credit default risk using financial and credit history features. Past due payment behavior, debt ratio and monthly income are useful indicators of credit risk.