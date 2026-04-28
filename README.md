
<div align="center">

# 💳 Credit Risk Intelligence Dashboard

### A Premium Machine Learning Dashboard for Credit Default Risk Prediction

**Developed by Wajiha Babar**

<br>

[![Python](https://img.shields.io/badge/Python-3.11-0A0A0A?style=for-the-badge&logo=python&logoColor=FFD700)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-0A0A0A?style=for-the-badge&logo=streamlit&logoColor=FFD700)]()
[![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML_Model-0A0A0A?style=for-the-badge&logo=scikitlearn&logoColor=FFD700)]()
[![Plotly](https://img.shields.io/badge/Plotly-Visualizations-0A0A0A?style=for-the-badge&logo=plotly&logoColor=FFD700)]()
[![Machine Learning](https://img.shields.io/badge/Machine_Learning-Credit_Risk-0A0A0A?style=for-the-badge&logoColor=FFD700)]()

<br>

🚀 **Live Dashboard:**  
[Credit Risk Prediction Dashboard](https://creditcard-risk-prediction.streamlit.app/)

</div>

---

## ✨ Project Overview

The **Credit Risk Intelligence Dashboard** is a premium machine learning web application designed to predict whether a loan applicant is likely to default or become seriously delinquent within two years.

This project follows a complete **end-to-end data science workflow**, including data loading, data cleaning, exploratory data analysis, model training, model evaluation, model saving, dashboard development, and live deployment.

The dashboard provides a luxury dark-gold interface where users can enter applicant details, generate credit risk predictions, view risk analytics, analyze important financial indicators, and download prediction reports as CSV files.

---

## 🎯 Objective

The main objective of this project is to predict credit default risk using applicant financial and credit history features.

The model predicts whether an applicant belongs to:

- **No Default Risk**
- **Default Risk**

This type of system can support financial institutions in understanding applicant risk patterns and making better data-driven lending decisions.

---

## 📊 Dataset

This project uses the **Kaggle Give Me Some Credit Dataset**.

The target column is:

```text
SeriousDlqin2yrs
```

### Target Meaning

```text
0 = Applicant did not default
1 = Applicant defaulted or became seriously delinquent
```

---

## 📌 Key Features Used

The dataset includes important credit risk indicators such as:

| Feature | Description |
|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | Credit utilization ratio |
| `age` | Applicant age |
| `NumberOfTime30-59DaysPastDueNotWorse` | Times applicant was 30-59 days late |
| `DebtRatio` | Debt to income ratio |
| `MonthlyIncome` | Monthly income of applicant |
| `NumberOfOpenCreditLinesAndLoans` | Number of active credit lines and loans |
| `NumberOfTimes90DaysLate` | Times applicant was 90 days late |
| `NumberRealEstateLoansOrLines` | Real estate loans or credit lines |
| `NumberOfTime60-89DaysPastDueNotWorse` | Times applicant was 60-89 days late |
| `NumberOfDependents` | Number of dependents |

---

## 🧠 Machine Learning Workflow

The complete project follows a professional data science pipeline:

```text
Data Loading
        ↓
Dataset Understanding
        ↓
Missing Value Handling
        ↓
Outlier Treatment
        ↓
Exploratory Data Analysis
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Best Model Selection
        ↓
Model Saving
        ↓
Streamlit Dashboard Deployment
```

---

## 🧹 Data Cleaning and Preparation

The following preprocessing steps were applied:

- Removed unnecessary index column
- Checked dataset shape, columns, and data types
- Identified missing values
- Handled missing values using **median imputation**
- Treated extreme values using **percentile-based outlier capping**
- Prepared features and target variable for machine learning model training

---

## 📈 Exploratory Data Analysis

Several visualizations were created to understand applicant behavior and credit risk patterns:

- Target distribution chart
- Monthly income distribution
- Age distribution by default status
- Debt ratio comparison
- Monthly income comparison
- Past due history comparison
- Correlation heatmap
- Feature importance chart
- ROC curve comparison
- Model performance comparison

These visualizations helped identify important patterns in applicant credit behavior and highlighted which financial features are more relevant for predicting default risk.

---

## 🤖 Machine Learning Models

Two classification models were trained and compared.

### 1. Logistic Regression

Logistic Regression was used as a simple, strong, and interpretable baseline classification model.

### 2. Decision Tree Classifier

Decision Tree Classifier was used to capture non-linear relationships between applicant financial behavior and credit default risk.

---

## 📏 Evaluation Metrics

The models were evaluated using multiple classification metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall correct predictions |
| Precision | Measures correct default predictions out of predicted defaults |
| Recall | Measures ability to identify actual default cases |
| F1 Score | Balances precision and recall |
| ROC-AUC | Measures model’s ability to separate risk classes |
| Confusion Matrix | Shows actual vs predicted classification results |
| MAE | Measures average prediction error |
| RMSE | Measures root mean squared prediction error |

---

## 💎 Dashboard Features

The deployed dashboard provides a luxury and professional interface with:

- Premium dark-gold theme
- Applicant credit risk prediction
- Predict button for user-controlled prediction
- Default probability gauge chart
- Risk category result
- Applicant input details
- Prediction probability donut chart
- Risk factor breakdown graph
- Applicant vs dataset median comparison
- Batch CSV prediction
- Downloadable single prediction CSV
- Downloadable batch prediction CSV
- Downloadable CSV template
- Professional executive-style dashboard layout

---

## 🚀 Live Deployment

The project is deployed using **Streamlit Cloud**.

🔗 **Live Dashboard:**  
[https://creditcard-risk-prediction.streamlit.app/](https://creditcard-risk-prediction.streamlit.app/)

---

## 🛠️ Tools and Technologies

| Category | Tools |
|---|---|
| Programming Language | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| Model Saving | Joblib |
| Dashboard | Streamlit |
| Development Environment | VS Code |
| Deployment | Streamlit Cloud |
| Version Control | Git, GitHub |

---

## 📁 Project Structure

```text
Credit Risk Prediction/
│
├── data/
│   └── raw/
│       ├── cs-training.csv
│       ├── cs-test.csv
│       ├── sampleEntry.csv
│       └── Data Dictionary.xls
│
├── notebooks/
│   └── Task2_Credit_Risk_Prediction.ipynb
│
├── models/
│   └── credit_risk_model.pkl
│
├── outputs/
│   ├── figures/
│   └── reports/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶️ How to Run This Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/Wajiha-Babar/Credit-Risk-Prediction.git
```

### 2. Move into the project folder

```bash
cd Credit-Risk-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

For Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

### 5. Install required libraries

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit dashboard

```bash
streamlit run app.py
```

---

## 📌 Results and Insights

The project shows that credit default risk is strongly influenced by:

- Past due payment history
- Debt ratio
- Credit utilization
- Monthly income
- Number of open credit lines
- Number of dependents

Applicants with more late payment records and higher debt-related values generally show higher default risk.

---

## 🏆 Conclusion

This project successfully demonstrates an end-to-end credit risk prediction system using machine learning.

The final solution includes:

- Clean data preprocessing
- Strong exploratory data analysis
- Classification model training
- Model evaluation
- Saved machine learning model
- Streamlit dashboard
- Live deployment

This project reflects practical skills in **data science, machine learning, visualization, model deployment, and dashboard development**.

---

## 👩‍💻 Author

**Wajiha Babar**  
Software Engineering Student | Data Science & Machine Learning Enthusiast  

GitHub: [Wajiha-Babar](https://github.com/Wajiha-Babar)  
LinkedIn: [Wajiha Babar](https://www.linkedin.com/in/wajiha-babar-12731a2bb/)

---

<div align="center">

### ⭐ If you like this project, feel free to star the repository.

**Credit Risk Intelligence Dashboard — Built with Python, Machine Learning and Streamlit**

</div>