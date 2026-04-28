import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Credit Risk Prediction Dashboard",
    page_icon="💳",
    layout="wide"
)

# Model path
MODEL_PATH = Path("models/credit_risk_model.pkl")

@st.cache_resource
def load_model():
    model_package = joblib.load(MODEL_PATH)
    return model_package

# Load model
model_package = load_model()
model = model_package["model"]
model_name = model_package["model_name"]
features = model_package["features"]

# Header
st.title("💳 Credit Risk Prediction Dashboard")

st.markdown("""
This dashboard predicts whether a loan applicant is likely to default based on financial and credit history details.
""")

st.info(f"Model Used: {model_name}")

# Sidebar
st.sidebar.header("Enter Applicant Details")

revolving_utilization = st.sidebar.slider(
    "Revolving Utilization of Unsecured Lines",
    min_value=0.0,
    max_value=2.0,
    value=0.45,
    step=0.01
)

age = st.sidebar.slider(
    "Age",
    min_value=18,
    max_value=100,
    value=35,
    step=1
)

late_30_59 = st.sidebar.slider(
    "Number of Times 30-59 Days Past Due",
    min_value=0,
    max_value=10,
    value=1,
    step=1
)

debt_ratio = st.sidebar.slider(
    "Debt Ratio",
    min_value=0.0,
    max_value=5.0,
    value=0.35,
    step=0.01
)

monthly_income = st.sidebar.number_input(
    "Monthly Income",
    min_value=0,
    max_value=1000000,
    value=5000,
    step=500
)

open_credit_lines = st.sidebar.slider(
    "Number of Open Credit Lines and Loans",
    min_value=0,
    max_value=50,
    value=8,
    step=1
)

late_90 = st.sidebar.slider(
    "Number of Times 90 Days Late",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

real_estate_loans = st.sidebar.slider(
    "Number of Real Estate Loans or Lines",
    min_value=0,
    max_value=20,
    value=1,
    step=1
)

late_60_89 = st.sidebar.slider(
    "Number of Times 60-89 Days Past Due",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

dependents = st.sidebar.slider(
    "Number of Dependents",
    min_value=0,
    max_value=20,
    value=2,
    step=1
)

# Input dataframe
input_data = pd.DataFrame([{
    "RevolvingUtilizationOfUnsecuredLines": revolving_utilization,
    "age": age,
    "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
    "DebtRatio": debt_ratio,
    "MonthlyIncome": monthly_income,
    "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
    "NumberOfTimes90DaysLate": late_90,
    "NumberRealEstateLoansOrLines": real_estate_loans,
    "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
    "NumberOfDependents": dependents
}])

# Correct feature order
input_data = input_data[features]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Applicant Input Data")
    st.dataframe(input_data, use_container_width=True)

if st.button("Predict Credit Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    risk_percentage = probability * 100

    with col2:
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("High Default Risk")
        else:
            st.success("Low Default Risk")

        st.metric("Default Probability", f"{risk_percentage:.2f}%")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_percentage,
            title={"text": "Credit Default Risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkred"},
                "steps": [
                    {"range": [0, 30], "color": "lightgreen"},
                    {"range": [30, 60], "color": "khaki"},
                    {"range": [60, 100], "color": "salmon"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": risk_percentage
                }
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk Interpretation")

    if risk_percentage < 30:
        st.write("The applicant appears to have a low credit default risk.")
    elif risk_percentage < 60:
        st.write("The applicant has a medium level of credit risk. Manual review is recommended.")
    else:
        st.write("The applicant has a high probability of default. Strong caution is recommended.")

st.markdown("---")
st.caption("Credit Risk Prediction Project | Built with Python, Scikit-learn and Streamlit")