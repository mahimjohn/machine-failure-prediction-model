import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)


# ---------------------------------------------------------
# Load Model and Scaler
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_model.joblib"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.joblib"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# ---------------------------------------------------------
# Feature Definitions
# ---------------------------------------------------------

numeric_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

feature_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_L",
    "Type_M"
]


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("⚙️ Predictive Maintenance Dashboard")

st.subheader("Industrial Equipment Health Monitoring")

st.write(
    "Enter the current equipment sensor readings to estimate "
    "failure risk and receive maintenance recommendations."
)


# ---------------------------------------------------------
# Equipment Inputs
# ---------------------------------------------------------

st.header("🔧 Equipment Sensor Readings")

col1, col2, col3 = st.columns(3)

with col1:
    machine_type = st.selectbox(
        "Machine Type",
        ["H", "M", "L"]
    )

    air_temperature = st.number_input(
        "Air Temperature [K]",
        min_value=295.0,
        max_value=305.0,
        value=300.0,
        step=0.1
    )

with col2:
    process_temperature = st.number_input(
        "Process Temperature [K]",
        min_value=305.0,
        max_value=315.0,
        value=310.0,
        step=0.1
    )

    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=1000,
        max_value=3000,
        value=1500,
        step=10
    )

with col3:
    torque = st.number_input(
        "Torque [Nm]",
        min_value=10.0,
        max_value=80.0,
        value=40.0,
        step=0.1
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0,
        max_value=300,
        value=100,
        step=1
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if st.button("🔍 Analyze Equipment", use_container_width=True):

    # Create input dataframe
    input_df = pd.DataFrame([{
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Type_L": 1 if machine_type == "L" else 0,
        "Type_M": 1 if machine_type == "M" else 0
    }])

    # Scale numerical features using the saved scaler
    input_scaled = input_df.copy()

    input_scaled[numeric_features] = scaler.transform(
        input_scaled[numeric_features]
    )

    # Ensure exact feature order
    input_scaled = input_scaled[feature_columns]

    # Predict
    failure_probability = model.predict_proba(
        input_scaled
    )[0, 1]

    predicted_failure = model.predict(
        input_scaled
    )[0]

    # -----------------------------------------------------
    # Risk Classification
    # -----------------------------------------------------

    if failure_probability >= 0.50:
        risk_level = "HIGH"
        recommendation = (
            "Immediate maintenance inspection recommended. "
            "Consider reducing equipment load if operational "
            "procedures permit."
        )

    elif failure_probability >= 0.20:
        risk_level = "MEDIUM"
        recommendation = (
            "Schedule a maintenance inspection and continue "
            "closely monitoring equipment conditions."
        )

    else:
        risk_level = "LOW"
        recommendation = "Continue normal monitoring."


    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    st.divider()

    st.header("📊 Equipment Health Assessment")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            "Failure Probability",
            f"{failure_probability:.2%}"
        )

    with result_col2:
        st.metric(
            "Predicted Failure",
            "YES" if predicted_failure == 1 else "NO"
        )

    with result_col3:
        st.metric(
            "Risk Level",
            risk_level
        )


    # -----------------------------------------------------
    # Risk Message
    # -----------------------------------------------------

    if risk_level == "HIGH":

        st.error(
            "🚨 HIGH FAILURE RISK\n\n"
            + recommendation
        )

    elif risk_level == "MEDIUM":

        st.warning(
            "⚠️ MEDIUM FAILURE RISK\n\n"
            + recommendation
        )

    else:

        st.success(
            "✅ LOW FAILURE RISK\n\n"
            + recommendation
        )


    # -----------------------------------------------------
    # Sensor Summary
    # -----------------------------------------------------

    st.subheader("Current Sensor Readings")

    display_data = pd.DataFrame({
        "Sensor": [
            "Machine Type",
            "Air Temperature [K]",
            "Process Temperature [K]",
            "Rotational Speed [rpm]",
            "Torque [Nm]",
            "Tool Wear [min]"
        ],
        "Current Value": [
            machine_type,
            air_temperature,
            process_temperature,
            rotational_speed,
            torque,
            tool_wear
        ]
    })

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )
    
    st.divider()

st.header("📊 Model Insights")

st.subheader("Feature Importance")

if hasattr(model, "feature_importances_"):

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        importance_df.set_index("Feature")
    )

else:
    st.info("Feature importance is not available for this model.")