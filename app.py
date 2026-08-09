import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Predictor | A.Masmi",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("churn_best_model.joblib")


@st.cache_data
def load_data():
    return pd.read_csv("telco_customer_churn_clean.csv")


@st.cache_data
def load_results():
    return pd.read_csv("model_comparison.csv")


model = load_model()
df = load_data()
results = load_results()

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 4rem;
    padding-bottom: 4rem;
}

.main-title {
    font-size: 3.2rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}

.subtitle {
    font-size: 1.15rem;
    color: #94a3b8;
    margin-bottom: 2rem;
}

.risk-high {
    padding: 20px;
    border-radius: 12px;
    border: 2px solid #ef4444;
}

.risk-medium {
    padding: 20px;
    border-radius: 12px;
    border: 2px solid #f59e0b;
}

.risk-low {
    padding: 20px;
    border-radius: 12px;
    border: 2px solid #22c55e;
}

.model-info {
    padding: 12px 16px;
    border: 1px solid #334155;
    border-radius: 10px;
    margin-top: 15px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Customer Churn Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Interactive Machine Learning Demo • A.Masmi</div>',
    unsafe_allow_html=True
)

st.write(
    """
    This application predicts the probability that a telecom customer
    will leave the company. The project uses a realistic customer churn
    dataset and compares three machine-learning algorithms.
    """
)

# ============================================================
# TOP METRICS
# ============================================================

best_row = results.sort_values(
    "ROC_AUC",
    ascending=False
).iloc[0]

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Customers",
    f"{len(df):,}"
)

c2.metric(
    "Models Compared",
    "3"
)

c3.metric(
    "Best ROC-AUC",
    f"{best_row['ROC_AUC']:.3f}"
)

c4.metric(
    "Best Model",
    best_row["Model"]
)

st.divider()

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Live Prediction",
    "Model Comparison",
    "Customer Insights",
    "About Project"
])

# ============================================================
# TAB 1 — LIVE PREDICTION
# ============================================================

with tab1:

    st.header("Try the Model")

    st.write(
        """
        Enter customer information below and let the trained
        Gradient Boosting model estimate the customer's churn probability.
        """
    )

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # COLUMN 1
    # --------------------------------------------------------

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure (months)",
            0,
            72,
            12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "No",
                "Yes",
                "No phone service"
            ]
        )

    # --------------------------------------------------------
    # COLUMN 2
    # --------------------------------------------------------

    with col2:

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        online_security = st.selectbox(
            "Online Security",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        device_protection = st.selectbox(
            "Device Protection",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

    # --------------------------------------------------------
    # COLUMN 3
    # --------------------------------------------------------

    with col3:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.slider(
            "Monthly Charges ($)",
            18.0,
            120.0,
            70.0,
            1.0
        )

        total_default = (
            monthly_charges * max(tenure, 1)
        )

        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=float(total_default),
            step=50.0
        )

    st.write("")

    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    predict_button = st.button(
        "Predict Churn Risk",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        customer = pd.DataFrame([{

            "gender":
                gender,

            "SeniorCitizen":
                1 if senior == "Yes" else 0,

            "Partner":
                partner,

            "Dependents":
                dependents,

            "tenure":
                tenure,

            "PhoneService":
                phone_service,

            "MultipleLines":
                multiple_lines,

            "InternetService":
                internet_service,

            "OnlineSecurity":
                online_security,

            "OnlineBackup":
                online_backup,

            "DeviceProtection":
                device_protection,

            "TechSupport":
                tech_support,

            "StreamingTV":
                streaming_tv,

            "StreamingMovies":
                streaming_movies,

            "Contract":
                contract,

            "PaperlessBilling":
                paperless,

            "PaymentMethod":
                payment_method,

            "MonthlyCharges":
                monthly_charges,

            "TotalCharges":
                total_charges

        }])

        probability = (
            model.predict_proba(customer)[0][1]
        )

        # ====================================================
        # RISK CLASSIFICATION
        # ====================================================

        if probability >= 0.70:

            risk = "HIGH RISK"
            risk_class = "risk-high"

        elif probability >= 0.40:

            risk = "MEDIUM RISK"
            risk_class = "risk-medium"

        else:

            risk = "LOW RISK"
            risk_class = "risk-low"

        st.divider()

        result_col1, result_col2 = st.columns(
            [1, 1]
        )

        # ====================================================
        # RESULT CARD
        # ====================================================

        with result_col1:

            st.markdown(
                f"""
                <div class="{risk_class}">
                    <h2>{risk}</h2>
                    <h1>{probability * 100:.1f}%</h1>
                    <p>
                    Estimated probability of customer churn
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            if risk == "HIGH RISK":

                st.warning(
                    """
                    This customer has a high predicted
                    likelihood of leaving. Retention action
                    may be appropriate.
                    """
                )

            elif risk == "MEDIUM RISK":

                st.info(
                    """
                    This customer shows moderate churn risk.
                    Monitoring or targeted retention may help.
                    """
                )

            else:

                st.success(
                    """
                    This customer currently has a relatively
                    low predicted churn risk.
                    """
                )

            st.markdown(
                f"""
                <div class="model-info">
                <b>Prediction model:</b> Gradient Boosting<br>
                <b>Test ROC-AUC:</b> {best_row['ROC_AUC']:.3f}
                </div>
                """,
                unsafe_allow_html=True
            )

        # ====================================================
        # IMPROVED GAUGE
        # ====================================================

        with result_col2:

            gauge = go.Figure(

                go.Indicator(

                    mode="gauge+number",

                    value=probability * 100,

                    number={
                        "suffix": "%",
                        "font": {
                            "size": 65
                        }
                    },

                    title={
                        "text":
                        "Churn Probability"
                    },

                    gauge={

                        "axis": {
                            "range": [0, 100],
                            "tickwidth": 1
                        },

                        "bar": {
                            "color":
                            "rgba(255,255,255,0.15)"
                        },

                        "steps": [

                            {
                                "range": [0, 40],
                                "color": "#16a34a"
                            },

                            {
                                "range": [40, 70],
                                "color": "#f59e0b"
                            },

                            {
                                "range": [70, 100],
                                "color": "#dc2626"
                            }

                        ],

                        "threshold": {

                            "line": {
                                "color": "white",
                                "width": 6
                            },

                            "thickness": 0.8,

                            "value":
                                probability * 100
                        }

                    }

                )

            )

            gauge.update_layout(

                height=350,

                margin=dict(
                    l=30,
                    r=30,
                    t=70,
                    b=20
                )

            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

        # ====================================================
        # CUSTOMER PROFILE
        # ====================================================

        st.subheader(
            "Customer Profile"
        )

        profile = pd.DataFrame({

            "Feature": [

                "Contract",
                "Tenure",
                "Internet Service",
                "Tech Support",
                "Monthly Charges",
                "Total Charges",
                "Payment Method"

            ],

            "Value": [

                contract,
                f"{tenure} months",
                internet_service,
                tech_support,
                f"${monthly_charges:.2f}",
                f"${total_charges:.2f}",
                payment_method

            ]

        })

        st.dataframe(
            profile,
            use_container_width=True,
            hide_index=True
        )

# ============================================================
# TAB 2 — MODEL COMPARISON
# ============================================================

with tab2:

    st.header(
        "Machine Learning Model Comparison"
    )

    st.write(
        """
        Three supervised classification models were trained
        and evaluated using the same train/test split.
        """
    )

    display_results = results.copy()

    for column in [

        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC"

    ]:

        display_results[column] = (
            display_results[column] * 100
        ).round(1)

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

    metric_long = results.melt(

        id_vars="Model",

        value_vars=[

            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC"

        ],

        var_name="Metric",
        value_name="Score"

    )

    fig = px.bar(

        metric_long,

        x="Model",
        y="Score",
        color="Metric",

        barmode="group",

        title=
        "Model Performance Comparison"

    )

    fig.update_yaxes(
        range=[0, 1],
        tickformat=".0%"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # ROC-AUC GRAPH
    # ========================================================

    roc_fig = px.bar(

        results.sort_values(
            "ROC_AUC",
            ascending=True
        ),

        x="ROC_AUC",
        y="Model",

        orientation="h",

        text="ROC_AUC",

        title=
        "ROC-AUC Ranking"

    )

    roc_fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    roc_fig.update_xaxes(
        range=[0, 1]
    )

    st.plotly_chart(
        roc_fig,
        use_container_width=True
    )

    st.subheader(
        "What the Results Show"
    )

    st.markdown(
        """
        **Gradient Boosting** achieved the highest overall
        ROC-AUC and accuracy.

        **Logistic Regression** achieved the highest recall,
        identifying a larger proportion of customers who
        actually churned.

        **Random Forest** produced the strongest F1 score
        in this experiment.

        This demonstrates why machine-learning model
        selection should not be based on accuracy alone.
        """
    )

# ============================================================
# TAB 3 — CUSTOMER INSIGHTS
# ============================================================

with tab3:

    st.header(
        "Customer Churn Insights"
    )

    chart_df = df.copy()

    chart_df["Churn Status"] = (
        chart_df["Churn"].map({
            0: "Stayed",
            1: "Churned"
        })
    )

    # ========================================================
    # OVERALL CHURN
    # ========================================================

    churn_counts = (

        chart_df[
            "Churn Status"
        ]

        .value_counts()

        .reset_index()

    )

    churn_counts.columns = [
        "Status",
        "Customers"
    ]

    fig1 = px.pie(

        churn_counts,

        names="Status",
        values="Customers",

        hole=0.45,

        title=
        "Overall Customer Churn"

    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # ========================================================
    # CONTRACT + INTERNET
    # ========================================================

    colA, colB = st.columns(2)

    with colA:

        contract_summary = (

            chart_df.groupby(
                "Contract",
                as_index=False
            )["Churn"]

            .mean()

        )

        contract_summary[
            "Churn Rate"
        ] = (
            contract_summary["Churn"] * 100
        )

        fig2 = px.bar(

            contract_summary,

            x="Contract",
            y="Churn Rate",

            title=
            "Churn Rate by Contract Type"

        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    with colB:

        internet_summary = (

            chart_df.groupby(
                "InternetService",
                as_index=False
            )["Churn"]

            .mean()

        )

        internet_summary[
            "Churn Rate"
        ] = (
            internet_summary["Churn"] * 100
        )

        fig3 = px.bar(

            internet_summary,

            x="InternetService",
            y="Churn Rate",

            title=
            "Churn Rate by Internet Service"

        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    # ========================================================
    # MONTHLY CHARGES + TENURE
    # ========================================================

    colC, colD = st.columns(2)

    with colC:

        fig4 = px.box(

            chart_df,

            x="Churn Status",
            y="MonthlyCharges",

            title=
            "Monthly Charges vs Churn"

        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )

    with colD:

        fig5 = px.box(

            chart_df,

            x="Churn Status",
            y="tenure",

            title=
            "Customer Tenure vs Churn"

        )

        st.plotly_chart(
            fig5,
            use_container_width=True
        )

    # ========================================================
    # INTERACTIVE DATASET
    # ========================================================

    st.subheader(
        "Explore the Dataset"
    )

    selected_contract = st.selectbox(

        "Filter by contract",

        ["All"] + sorted(
            chart_df[
                "Contract"
            ].unique().tolist()
        )

    )

    if selected_contract == "All":

        filtered = chart_df

    else:

        filtered = chart_df[
            chart_df["Contract"]
            == selected_contract
        ]

    st.write(
        f"Customers shown: {len(filtered):,}"
    )

    st.dataframe(
        filtered.head(100),
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# TAB 4 — ABOUT PROJECT
# ============================================================

with tab4:

    st.header(
        "About This Project"
    )

    st.markdown(
        """
        ### Project Goal

        Build an end-to-end machine-learning application
        for predicting telecom customer churn and make the
        trained model accessible through an interactive web
        application.

        ### Machine Learning Workflow

        1. Load customer data
        2. Clean and prepare the dataset
        3. Explore customer churn patterns
        4. Encode categorical variables
        5. Scale numerical variables
        6. Train three classification models
        7. Compare multiple evaluation metrics
        8. Select the strongest overall model
        9. Save the trained ML pipeline
        10. Deploy the model with Streamlit

        ### Models Compared

        - Logistic Regression
        - Random Forest
        - Gradient Boosting

        ### Technologies

        **Python**

        **pandas**

        **scikit-learn**

        **JupyterLab**

        **Plotly**

        **Streamlit**

        **Git & GitHub**

        ### Evaluation Metrics

        The project evaluates models using:

        - Accuracy
        - Precision
        - Recall
        - F1-score
        - ROC-AUC

        Multiple metrics are used because different models
        can perform better on different aspects of the
        classification problem.

        ### Portfolio Project

        Developed by **A.Masmi** as an interactive
        Machine Learning and Data Science portfolio project.
        """
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#94a3b8;
        padding:10px;
    ">
        A.Masmi • Machine Learning & Data Science Portfolio
    </div>
    """,
    unsafe_allow_html=True
)
