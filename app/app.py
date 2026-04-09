# ============================================================
# Job Salary Predictor — Streamlit App
# Author: Rohith Phani Gavirneni
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# --- Page config ---
st.set_page_config(
    page_title="Data Science Salary Predictor",
    page_icon="💼",
    layout="centered"
)

# --- Load model and features ---
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'salary_model.pkl')
    columns_path = os.path.join(base_dir, 'models', 'feature_columns.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(columns_path, 'rb') as f:
        columns = pickle.load(f)
    return model, columns

model, feature_columns = load_model()

# --- Header ---
st.title("💼 Data Science Salary Predictor")
st.markdown("Predict your expected salary based on role, experience, and location.")
st.markdown("---")

# --- Input form ---
col1, col2 = st.columns(2)

with col1:
    experience = st.selectbox(
        "Experience Level",
        options=['Entry-level', 'Mid-level', 'Senior', 'Executive']
    )
    job_title = st.selectbox(
        "Job Title",
        options=[
            'Data Scientist', 'Data Engineer', 'Data Analyst',
            'Machine Learning Engineer', 'Research Scientist',
            'Data Science Manager', 'Data Architect',
            'Machine Learning Scientist', 'Big Data Engineer',
            'Director of Data Science', 'Other'
        ]
    )
    company_size = st.selectbox(
        "Company Size",
        options=['Small', 'Medium', 'Large']
    )

with col2:
    is_us_resident = st.selectbox(
        "Employee Location",
        options=['US-based', 'Outside US']
    )
    is_us_company = st.selectbox(
        "Company Location",
        options=['US Company', 'Non-US Company']
    )
    remote_ratio = st.selectbox(
        "Remote Work Policy",
        options=['On-site (0%)', 'Hybrid (50%)', 'Remote (100%)']
    )

work_year = st.slider("Work Year", min_value=2020, max_value=2022, value=2022)
st.markdown("---")

# --- Encode inputs ---
exp_map    = {'Entry-level': 1, 'Mid-level': 2, 'Senior': 3, 'Executive': 4}
size_map   = {'Small': 1, 'Medium': 2, 'Large': 3}
remote_map = {'On-site (0%)': 0, 'Hybrid (50%)': 50, 'Remote (100%)': 100}

# Build base input dict
input_data = {col: 0 for col in feature_columns}
input_data['experience_encoded']    = exp_map[experience]
input_data['is_us_company']         = 1 if is_us_company == 'US Company' else 0
input_data['is_us_resident']        = 1 if is_us_resident == 'US-based' else 0
input_data['company_size_encoded']  = size_map[company_size]
input_data['remote_ratio']          = remote_map[remote_ratio]
input_data['is_fulltime']           = 1
input_data['work_year']             = work_year

# One-hot encode job title
title_col = f'job_title_grouped_{job_title}'
if title_col in input_data:
    input_data[title_col] = 1

input_df = pd.DataFrame([input_data])[feature_columns]

# --- Predict ---
if st.button("🔮 Predict Salary", use_container_width=True):
    log_pred         = model.predict(input_df)[0]
    predicted_salary = np.expm1(log_pred)
    mae              = 29360
    low              = max(0, predicted_salary - mae)
    high             = predicted_salary + mae

    st.success(f"### 💰 Estimated Salary: ${predicted_salary:,.0f} USD/year")
    st.info(f"📊 Likely Range: ${low:,.0f} — ${high:,.0f}  *(based on model MAE of $29,360)*")

    st.markdown("---")
    st.markdown("**Key factors in this prediction:**")
    st.markdown(f"- Location: {'🇺🇸 US-based' if is_us_resident == 'US-based' else '🌍 Outside US'} *(strongest predictor)*")
    st.markdown(f"- Experience: {experience}")
    st.markdown(f"- Company: {is_us_company}, {company_size} size")

# --- Footer ---
st.markdown("---")
st.caption("Model: Random Forest | R²: 0.548 | MAE: $29,360 | Dataset: 607 Data Science roles (2020–2022)")