# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:49:44 2025

@author: LENOVO
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 13 13:48:35 2025

@author: LENOVO
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.linear_model import LogisticRegression
import os

# ——— Page configuration ———
st.set_page_config(
    page_title="PCNL Post-Operative Fever Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ——— Custom CSS ———
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
    }
    h1 { color: #2C3E50; }
    h2 { color: #3498DB; }
    .footer {
        margin-top: 2rem;
        text-align: center;
        color: #7F8C8D;
    }
</style>
""", unsafe_allow_html=True)

# ——— Load model with caching ———
@st.cache_resource
def load_model():
    try:
        return joblib.load("LR.pkl")
    except FileNotFoundError:
        st.error("Model file 'LR.pkl' not found. Please place it alongside this script.")
        return None

# ——— Title and description ———
st.title("PCNL Post-Operative Fever Prediction")
st.markdown("### A machine learning–based tool to estimate fever risk after PCNL")

# ——— Sidebar ———
with st.sidebar:
    st.header("About the Model")
    st.info(
        """
        This logistic regression model is trained on historical clinical data
        to predict the risk of post-operative fever after percutaneous nephrolithotomy (PCNL).
        Enter patient parameters on the main page and click "Predict Fever Risk."
        """
    )
    st.header("Feature Descriptions")
    st.markdown("""
    - **LMR**: Lymphocyte-to-Monocyte Ratio  
    - **PLR**: Platelet-to-Lymphocyte Ratio  
    - **BMI**: Body Mass Index  
    - **Mayo Score**: Mayo Surgical Complexity Score for PCNL procedures
      - Scores < 3: Lower surgical complexity
      - Scores ≥ 3: Higher surgical complexity
    """)

# ——— Feature configuration ———
feature_ranges = {
    "LMR": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 5.0, "description": "Lymphocyte-to-Monocyte Ratio"},
    "Preoperative_N": {"type": "numerical", "min": 0.0, "max": 30.0, "default": 4.0, "description": "Preoperative Neutrophil Count (×10⁹/L)"},
    "Operative_time": {"type": "numerical", "min": 10, "max": 300, "default": 60, "description": "Operative Time (minutes)"},
    "Preoperative_WBC": {"type": "numerical", "min": 0.0, "max": 30.0, "default": 7.0, "description": "Preoperative WBC (×10⁹/L)"},
    "Preoperative_L": {"type": "numerical", "min": 0.0, "max": 10.0, "default": 1.8, "description": "Preoperative Lymphocyte Count (×10⁹/L)"},
    "PLR": {"type": "numerical", "min": 0.0, "max": 1000.0, "default": 120.0, "description": "Platelet-to-Lymphocyte Ratio"},
    "Preoperative_hemoglobin": {"type": "numerical", "min": 50.0, "max": 200.0, "default": 130.0, "description": "Preoperative Hemoglobin (g/L)"},
    "Number_of_stones": {"type": "numerical", "min": 1, "max": 20, "default": 1, "description": "Number of Stones"},
    "BMI": {"type": "numerical", "min": 10.0, "max": 50.0, "default": 24.0, "description": "Body Mass Index (kg/m²)"},
    "Sex": {"type": "categorical", "options": ["Male", "Female"], "default": "Male", "description": "Sex"},
    "Diabetes_mellitus": {"type": "categorical", "options": ["No", "Yes"], "default": "No", "description": "Diabetes Mellitus"},
    "UrineLeuk_bin": {"type": "categorical", "options": ["=0", ">0"], "default": "=0", "description": "Urine Leukocytes"},
    "Channel_size": {"type": "categorical", "options": ["18F", "20F"], "default": "18F", "description": "Channel Size"},
    "degree_of_hydronephrosis": {
        "type": "categorical",
        "options": ["None", "Mild", "Moderate", "Severe"],
        "default": "None",
        "description": "Degree of Hydronephrosis"
    },
    "MayoScore_bin": {"type": "categorical", "options": ["<3", "≥3"], "default": "<3", "description": "Mayo Score"}
}

# ——— Input form ———
st.header("Enter Patient Parameters")
cols = st.columns(3)
input_data = {}
for idx, (feat, cfg) in enumerate(feature_ranges.items()):
    col = cols[idx % 3]
    with col:
        if cfg["type"] == "numerical":
            input_data[feat] = st.number_input(
                label=f"{cfg['description']} ({feat})",
                min_value=cfg["min"],
                max_value=cfg["max"],
                value=cfg["default"],
                help=f"Allowed range: {cfg['min']} to {cfg['max']}"
            )
        else:
            input_data[feat] = st.selectbox(
                label=f"{cfg['description']} ({feat})",
