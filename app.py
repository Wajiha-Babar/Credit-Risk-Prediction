import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Credit Risk Intelligence Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM PREMIUM CSS
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #070707 0%, #111111 45%, #1c1208 100%);
        color: #f5f5f5;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070707 0%, #151515 100%);
        border-right: 1px solid rgba(212, 175, 55, 0.35);
    }

    .main-title {
        font-size: 44px;
        font-weight: 900;
        color: #f7d774;
        letter-spacing: 0.5px;
        margin-bottom: 0px;
    }

    .sub-title {
        font-size: 18px;
        color: #d7d7d7;
        margin-top: 0px;
        margin-bottom: 25px;
    }

    .info-card {
        background: rgba(255, 255, 255, 0.055);
        border: 1px solid rgba(212, 175, 55, 0.28);
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 12px 34px rgba(0, 0, 0, 0.45);
        margin-bottom: 20px;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(212,175,55,0.20), rgba(255,255,255,0.06));
        border: 1px solid rgba(212, 175, 55, 0.42);
        border-radius: 22px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 10px 28px rgba(0,0,0,0.40);
        min-height: 145px;
    }

    .metric-value {
        font-size: 34px;
        font-weight: 900;
        color: #f7d774;
        margin: 0px;
        line-height: 1.2;
    }

    .metric-label {
        font-size: 14px;
        color: #d0d0d0;
        margin-top: 8px;
    }

    .risk-low {
        color: #37d67a;
        font-size: 28px;
        font-weight: 900;
        margin: 0px;
    }

    .risk-medium {
        color: #f7d774;
        font-size: 28px;
        font-weight: 900;
        margin: 0px;
    }

    .risk-high {
        color: #ff4b4b;
        font-size: 28px;
        font-weight: 900;
        margin: 0px;
    }

    .recommend-card {
        background: linear-gradient(135deg, rgba(212,175,55,0.16), rgba(255,255,255,0.045));
        border: 1px solid rgba(212,175,55,0.35);
        border-radius: 20px;
        padding: 22px;
        margin-bottom: 18px;
    }

    .section-heading {
        color: #f7d774;
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #f7d774, #b88a1d);
        color: #111111;
        border: none;
        border-radius: 16px;
        padding: 13px 26px;
        font-weight: 900;
        font-size: 17px;
        box-shadow: 0 8px 24px rgba(212,175,55,0.35);
        width: 100%;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #ffe58a, #d4af37);
        color: #000000;
        border: none;
    }

    div.stDownloadButton > button {
        background: linear-gradient(135deg, #f7d774, #b88a1d);
        color: #111111;
        border: none;
        border-radius: 14px;
        padding: 10px 22px;
        font-weight: 900;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.06);
        border-radius: 14px;
        color: #f7d774;
        padding: 12px 20px;
        border: 1px solid rgba(212,175,55,0.18);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(212,175,55,0.35), rgba(255,255,255,0.08));
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# PATHS
# =========================================================
MODEL_PATH = Path("models/credit_risk_model.pkl")
DATA_PATH = Path("data/raw/cs-training.csv")

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    model_package = joblib.load(MODEL_PATH)
    return model_package

# =========================================================
# LOAD DATASET FOR BENCHMARK GRAPHS
# =========================================================
@st.cache_data
def load_dataset():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)

        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        if "MonthlyIncome" in df.columns:
            df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())

        if "NumberOfDependents" in df.columns:
            df["NumberOfDependents"] = df["NumberOfDependents"].fillna(df["NumberOfDependents"].median())

        return df

    return None

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def risk_category(probability):
    risk_percentage = probability * 100

    if risk_percentage < 30:
        return "Low Risk", "risk-low", "Approved / Safe Candidate"
    elif risk_percentage < 60:
        return "Medium Risk", "risk-medium", "Manual Review Recommended"
    else:
        return "High Risk", "risk-high", "Strong Caution Required"


def create_input_dataframe(
    revolving_utilization,
    age,
    late_30_59,
    debt_ratio,
    monthly_income,
    open_credit_lines,
    late_90,
    real_estate_loans,
    late_60_89,
    dependents,
    features
):
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

    input_data = input_data[features]
    return input_data


def get_risk_factor_scores(input_data, dataset):
    row = input_data.iloc[0]

    income_median = 5000
    if dataset is not None and "MonthlyIncome" in dataset.columns:
        income_median = dataset["MonthlyIncome"].median()

    if income_median == 0:
        income_median = 1

    income_pressure = max(0, (income_median - row["MonthlyIncome"]) / income_median) * 100

    factors = pd.DataFrame({
        "Risk Factor": [
            "Credit Utilization",
            "Debt Ratio",
            "30-59 Days Late",
            "60-89 Days Late",
            "90 Days Late",
            "Income Pressure",
            "Dependents Load"
        ],
        "Risk Score": [
            min(row["RevolvingUtilizationOfUnsecuredLines"] / 1.5, 1) * 100,
            min(row["DebtRatio"] / 2.0, 1) * 100,
            min(row["NumberOfTime30-59DaysPastDueNotWorse"] / 5, 1) * 100,
            min(row["NumberOfTime60-89DaysPastDueNotWorse"] / 5, 1) * 100,
            min(row["NumberOfTimes90DaysLate"] / 5, 1) * 100,
            min(income_pressure, 100),
            min(row["NumberOfDependents"] / 6, 1) * 100
        ]
    })

    return factors.sort_values(by="Risk Score", ascending=True)


def create_gauge_chart(risk_percentage):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_percentage,
        number={"suffix": "%", "font": {"size": 42, "color": "#f7d774"}},
        title={"text": "Default Risk Probability", "font": {"size": 21, "color": "#ffffff"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#ffffff"},
            "bar": {"color": "#d4af37"},
            "bgcolor": "rgba(255,255,255,0.05)",
            "borderwidth": 1,
            "bordercolor": "rgba(212,175,55,0.4)",
            "steps": [
                {"range": [0, 30], "color": "rgba(55,214,122,0.35)"},
                {"range": [30, 60], "color": "rgba(247,215,116,0.35)"},
                {"range": [60, 100], "color": "rgba(255,75,75,0.35)"}
            ],
            "threshold": {
                "line": {"color": "#ff4b4b", "width": 5},
                "thickness": 0.8,
                "value": risk_percentage
            }
        }
    ))

    fig.update_layout(
        height=390,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ffffff"},
        margin=dict(l=20, r=20, t=55, b=20)
    )

    return fig


def create_probability_donut(default_probability):
    non_default_probability = 1 - default_probability

    fig = go.Figure(data=[
        go.Pie(
            labels=["No Default Probability", "Default Probability"],
            values=[non_default_probability * 100, default_probability * 100],
            hole=0.62,
            marker=dict(colors=["#37d67a", "#ff4b4b"]),
            textinfo="label+percent",
            textfont=dict(color="#ffffff", size=13)
        )
    ])

    fig.update_layout(
        title="Prediction Probability Split",
        title_font=dict(color="#ffffff", size=21),
        height=390,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff"),
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=True,
        legend=dict(font=dict(color="#ffffff"))
    )

    return fig


def create_risk_factor_chart(risk_factors):
    fig = px.bar(
        risk_factors,
        x="Risk Score",
        y="Risk Factor",
        orientation="h",
        text=risk_factors["Risk Score"].round(1),
        color="Risk Score",
        color_continuous_scale=["#37d67a", "#f7d774", "#ff4b4b"]
    )

    fig.update_traces(
        texttemplate="%{text}%",
        textposition="outside"
    )

    fig.update_layout(
        title="Applicant Risk Factor Breakdown",
        title_font=dict(color="#ffffff", size=21),
        xaxis_title="Risk Strength",
        yaxis_title="",
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#ffffff"),
        coloraxis_showscale=False,
        margin=dict(l=20, r=60, t=60, b=20)
    )

    return fig


def create_benchmark_chart(input_data, dataset):
    if dataset is None:
        return None

    selected_features = [
        "RevolvingUtilizationOfUnsecuredLines",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberOfDependents"
    ]

    benchmark_rows = []

    for feature in selected_features:
        if feature in dataset.columns and feature in input_data.columns:
            applicant_value = input_data.iloc[0][feature]
            dataset_median = dataset[feature].median()

            benchmark_rows.append({
                "Feature": feature,
                "Applicant": applicant_value,
                "Dataset Median": dataset_median
            })

    benchmark_df = pd.DataFrame(benchmark_rows)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=benchmark_df["Feature"],
        y=benchmark_df["Applicant"],
        name="Applicant",
        marker_color="#d4af37"
    ))

    fig.add_trace(go.Bar(
        x=benchmark_df["Feature"],
        y=benchmark_df["Dataset Median"],
        name="Dataset Median",
        marker_color="#6c757d"
    ))

    fig.update_layout(
        title="Applicant vs Dataset Median",
        title_font=dict(color="#ffffff", size=21),
        xaxis_title="Features",
        yaxis_title="Value",
        barmode="group",
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#ffffff"),
        margin=dict(l=20, r=20, t=60, b=110),
        legend=dict(font=dict(color="#ffffff"))
    )

    fig.update_xaxes(tickangle=35)

    return fig


def create_csv_download_data(input_data, prediction, probability, category, recommendation, model_name):
    export_df = input_data.copy()
    export_df["Prediction"] = "Default Risk" if prediction == 1 else "No Default Risk"
    export_df["Default Probability (%)"] = round(probability * 100, 2)
    export_df["Risk Category"] = category
    export_df["Recommendation"] = recommendation
    export_df["Model Used"] = model_name
    export_df["Prediction Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return export_df


# =========================================================
# LOAD MODEL AND DATA
# =========================================================
if not MODEL_PATH.exists():
    st.error("Model file not found. Please run the notebook first and save credit_risk_model.pkl inside the models folder.")
    st.stop()

model_package = load_model()
model = model_package["model"]
model_name = model_package["model_name"]
features = model_package["features"]

dataset = load_dataset()

# =========================================================
# SESSION STATE FOR PREDICTION BUTTON
# =========================================================
if "show_prediction" not in st.session_state:
    st.session_state.show_prediction = False

# =========================================================
# HEADER
# =========================================================
st.markdown('<p class="main-title">💳 Credit Risk Intelligence Dashboard</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Premium AI-powered dashboard for credit default risk prediction, applicant scoring, risk insights, and downloadable reports.</p>',
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="info-card">
        <b>Model in Use:</b> {model_name}<br>
        <b>Purpose:</b> Predict whether an applicant is likely to default or become seriously delinquent within two years.
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR INPUTS
# =========================================================
st.sidebar.markdown("## Applicant Details")
st.sidebar.caption("Enter applicant values, then click Predict Credit Risk.")

revolving_utilization = st.sidebar.slider(
    "Revolving Utilization",
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
    "30-59 Days Past Due",
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
    "Open Credit Lines / Loans",
    min_value=0,
    max_value=50,
    value=8,
    step=1
)

late_90 = st.sidebar.slider(
    "90 Days Late",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

real_estate_loans = st.sidebar.slider(
    "Real Estate Loans / Lines",
    min_value=0,
    max_value=20,
    value=1,
    step=1
)

late_60_89 = st.sidebar.slider(
    "60-89 Days Past Due",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

dependents = st.sidebar.slider(
    "Dependents",
    min_value=0,
    max_value=20,
    value=2,
    step=1
)

sidebar_predict_button = st.sidebar.button("✨ Predict Credit Risk")

if sidebar_predict_button:
    st.session_state.show_prediction = True

# =========================================================
# INPUT DATA
# =========================================================
input_data = create_input_dataframe(
    revolving_utilization,
    age,
    late_30_59,
    debt_ratio,
    monthly_income,
    open_credit_lines,
    late_90,
    real_estate_loans,
    late_60_89,
    dependents,
    features
)

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "Executive Prediction",
    "Risk Analytics",
    "Batch CSV Prediction"
])

# =========================================================
# TAB 1: EXECUTIVE PREDICTION
# =========================================================
with tab1:
    main_predict_button = st.button("✨ Predict Credit Risk Now")

    if main_predict_button:
        st.session_state.show_prediction = True

    if not st.session_state.show_prediction:
        st.markdown(
            """
            <div class="info-card">
                <h3 style="color:#f7d774;">Ready for Prediction</h3>
                <p>Please enter applicant details from the left sidebar and click the <b>Predict Credit Risk</b> button.</p>
                <p>After prediction, this dashboard will show default probability, risk category, charts, applicant comparison and downloadable CSV report.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("### Current Applicant Input Preview")
        preview_df = input_data.T.reset_index()
        preview_df.columns = ["Feature", "Value"]
        st.dataframe(preview_df, use_container_width=True)

    else:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        risk_percentage = probability * 100

        category, category_class, recommendation = risk_category(probability)

        download_df = create_csv_download_data(
            input_data=input_data,
            prediction=prediction,
            probability=probability,
            category=category,
            recommendation=recommendation,
            model_name=model_name
        )

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <p class="metric-value">{risk_percentage:.2f}%</p>
                    <p class="metric-label">Default Probability</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with kpi2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <p class="{category_class}">{category}</p>
                    <p class="metric-label">Risk Category</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with kpi3:
            output_text = "Default" if prediction == 1 else "No Default"
            st.markdown(
                f"""
                <div class="metric-card">
                    <p class="metric-value">{output_text}</p>
                    <p class="metric-label">Model Output</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with kpi4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <p class="metric-value">{model_name}</p>
                    <p class="metric-label">ML Model</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        left, right = st.columns([1.15, 1])

        with left:
            st.plotly_chart(create_gauge_chart(risk_percentage), use_container_width=True)

        with right:
            st.markdown(
                f"""
                <div class="recommend-card">
                    <p class="{category_class}">{recommendation}</p>
                    <p style="color:#d7d7d7;">
                    This result is generated using applicant financial profile and credit history features.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("### Applicant Input Details")

            display_input = input_data.T.reset_index()
            display_input.columns = ["Feature", "Value"]
            st.dataframe(display_input, use_container_width=True)

            csv_data = download_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇ Download This Prediction as CSV",
                data=csv_data,
                file_name="single_applicant_credit_risk_prediction.csv",
                mime="text/csv"
            )

# =========================================================
# TAB 2: RISK ANALYTICS
# =========================================================
with tab2:
    if not st.session_state.show_prediction:
        st.markdown(
            """
            <div class="info-card">
                <h3 style="color:#f7d774;">Risk Analytics Locked</h3>
                <p>Please click <b>Predict Credit Risk</b> first to generate charts and risk analytics.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        risk_percentage = probability * 100

        risk_factors = get_risk_factor_scores(input_data, dataset)

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.plotly_chart(create_probability_donut(probability), use_container_width=True)

        with chart_col2:
            st.plotly_chart(create_risk_factor_chart(risk_factors), use_container_width=True)

        benchmark_fig = create_benchmark_chart(input_data, dataset)

        if benchmark_fig is not None:
            st.plotly_chart(benchmark_fig, use_container_width=True)

        st.write("### Risk Factor Table")
        st.dataframe(risk_factors.sort_values(by="Risk Score", ascending=False), use_container_width=True)

# =========================================================
# TAB 3: BATCH CSV PREDICTION
# =========================================================
with tab3:
    st.markdown(
        """
        <div class="info-card">
        <h3 style="color:#f7d774;">Batch Prediction Using CSV</h3>
        <p>Upload a CSV file with the same feature columns used in training. The dashboard will predict credit risk for every applicant.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    template_df = pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines": 0.45,
        "age": 35,
        "NumberOfTime30-59DaysPastDueNotWorse": 1,
        "DebtRatio": 0.35,
        "MonthlyIncome": 5000,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2
    }])[features]

    st.download_button(
        label="⬇ Download CSV Template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="credit_risk_prediction_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload applicant CSV file", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        if "Unnamed: 0" in batch_df.columns:
            batch_df = batch_df.drop(columns=["Unnamed: 0"])

        missing_columns = [col for col in features if col not in batch_df.columns]

        if missing_columns:
            st.error(f"Your uploaded CSV is missing these columns: {missing_columns}")
            st.write("Required columns are:")
            st.write(features)
        else:
            batch_input = batch_df[features].copy()

            batch_predictions = model.predict(batch_input)
            batch_probabilities = model.predict_proba(batch_input)[:, 1]

            result_df = batch_df.copy()
            result_df["Prediction"] = np.where(batch_predictions == 1, "Default Risk", "No Default Risk")
            result_df["Default Probability (%)"] = np.round(batch_probabilities * 100, 2)
            result_df["Risk Category"] = pd.cut(
                result_df["Default Probability (%)"],
                bins=[-1, 30, 60, 100],
                labels=["Low Risk", "Medium Risk", "High Risk"],
                include_lowest=True
            )
            result_df["Model Used"] = model_name
            result_df["Prediction Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            st.success("Batch prediction completed successfully.")

            b1, b2, b3, b4 = st.columns(4)

            with b1:
                st.metric("Total Applicants", len(result_df))

            with b2:
                high_risk_count = (result_df["Risk Category"] == "High Risk").sum()
                st.metric("High Risk Applicants", int(high_risk_count))

            with b3:
                medium_risk_count = (result_df["Risk Category"] == "Medium Risk").sum()
                st.metric("Medium Risk Applicants", int(medium_risk_count))

            with b4:
                avg_risk = result_df["Default Probability (%)"].mean()
                st.metric("Average Default Risk", f"{avg_risk:.2f}%")

            graph1, graph2 = st.columns(2)

            with graph1:
                risk_count = result_df["Risk Category"].value_counts().reset_index()
                risk_count.columns = ["Risk Category", "Count"]

                fig_batch_bar = px.bar(
                    risk_count,
                    x="Risk Category",
                    y="Count",
                    color="Risk Category",
                    color_discrete_map={
                        "Low Risk": "#37d67a",
                        "Medium Risk": "#f7d774",
                        "High Risk": "#ff4b4b"
                    },
                    text="Count",
                    title="Batch Risk Category Distribution"
                )

                fig_batch_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.03)",
                    font=dict(color="#ffffff"),
                    height=420
                )

                st.plotly_chart(fig_batch_bar, use_container_width=True)

            with graph2:
                fig_hist = px.histogram(
                    result_df,
                    x="Default Probability (%)",
                    nbins=30,
                    color="Risk Category",
                    color_discrete_map={
                        "Low Risk": "#37d67a",
                        "Medium Risk": "#f7d774",
                        "High Risk": "#ff4b4b"
                    },
                    title="Default Probability Distribution"
                )

                fig_hist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.03)",
                    font=dict(color="#ffffff"),
                    height=420
                )

                st.plotly_chart(fig_hist, use_container_width=True)

            st.write("### Batch Prediction Results")
            st.dataframe(result_df, use_container_width=True)

            batch_csv = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇ Download Full Batch Prediction CSV",
                data=batch_csv,
                file_name="batch_credit_risk_predictions.csv",
                mime="text/csv"
            )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("Credit Risk Intelligence Dashboard | Author: Wajiha Babar")