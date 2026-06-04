import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ─────────────────────────────────────────────────────────────
# Load Model and Feature Columns
# ─────────────────────────────────────────────────────────────

model = joblib.load("models/model.pkl")

feature_columns = joblib.load(

    "models/feature_columns.pkl"

)

# ─────────────────────────────────────────────────────────────
# Streamlit Page Config
# ─────────────────────────────────────────────────────────────

st.set_page_config(

    page_title="Student Performance Predictor",

    page_icon="🎓",

    layout="centered"

)

# ─────────────────────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────────────────────

st.title("🎓 Student Performance Predictor")

st.markdown(

    "Predict final student score using machine learning."

)

# ─────────────────────────────────────────────────────────────
# User Inputs
# ─────────────────────────────────────────────────────────────

study_hours = st.number_input(

    "Study Hours",

    min_value=0.0,

    max_value=24.0,

    value=5.0

)

attendance = st.number_input(

    "Attendance Percentage",

    min_value=0.0,

    max_value=100.0,

    value=80.0

)

assignments_completed = st.number_input(

    "Assignments Completed (%)",

    min_value=0.0,

    max_value=100.0,

    value=75.0

)

sleep_hours = st.number_input(

    "Sleep Hours",

    min_value=0.0,

    max_value=24.0,

    value=7.0

)

previous_gpa = st.number_input(

    "Previous GPA",

    min_value=0.0,

    max_value=10.0,

    value=7.5

)

internet_usage = st.number_input(

    "Internet Usage Hours",

    min_value=0.0,

    max_value=24.0,

    value=3.0

)

# ─────────────────────────────────────────────────────────────
# Create Input DataFrame
# ─────────────────────────────────────────────────────────────

input_data = pd.DataFrame(

    [[

        study_hours,
        attendance,
        assignments_completed,
        sleep_hours,
        previous_gpa,
        internet_usage

    ]],

    columns=[

        "study_hours",
        "attendance_percentage",
        "assignments_completed_percentage",
        "sleep_hours",
        "previous_gpa",
        "internet_usage_hours"

    ]

)

# ─────────────────────────────────────────────────────────────
# Align Features
# ─────────────────────────────────────────────────────────────

for column in feature_columns:

    if column not in input_data.columns:

        input_data[column] = 0

input_data = input_data[feature_columns]

# ─────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────

if st.button("Predict Final Score"):

    prediction = model.predict(input_data)[0]

    st.success(

        f"Predicted Final Score: {prediction:.2f}"

    )

    # Optional Interpretation

    if prediction >= 90:

        st.info("Excellent Performance 🚀")

    elif prediction >= 75:

        st.info("Good Performance 👍")

    elif prediction >= 50:

        st.warning("Average Performance ⚠️")

    else:

        st.error("Needs Improvement 📚")