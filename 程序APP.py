# -*- coding: utf-8 -*-
"""
Created on Tue May 13 11:08:19 2025

@author: LENOVO
"""

# app.py
# -*- coding: utf-8 -*-
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression

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
    # 从脚本同目录下加载 LR.pkl
    model_path = os.path.join(os.path.dirname(__file__), "LR.pkl")
    if not os.path.exists(model_path):
        st.error(f"模型文件未找到：{model_path}")
        return None
    return joblib.load(model_path)

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
    - **Mayo Score**: Mayo Surgical Complexity Score  
    """)

# ——— Feature configuration ———
feature_ranges = {
    "LMR": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 5.0, "description": "Lymphocyte-to-Monocyte Ratio"},
    "Preoperative_N": {"type": "numerical", "min": 0.0, "max": 30.0,   "default": 4.0, "description": "Preoperative Neutrophil Count (×10⁹/L)"},
    "Operative_time": {"type": "numerical", "min": 10,  "max": 300,     "default": 60,  "description": "Operative Time (minutes)"},
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
                options=cfg["options"],
                index=cfg["options"].index(cfg["default"])
            )

st.markdown("---")
if st.button("Predict Fever Risk", use_container_width=True):
    model = load_model()
    if model is None:
        st.stop()

    # 准备 DataFrame
    df = pd.DataFrame([input_data])

    # 编码
    df["Sex"] = df["Sex"].map({"Male": 1, "Female": 0})
    df["Diabetes_mellitus"] = df["Diabetes_mellitus"].map({"Yes": 1, "No": 0})
    df["UrineLeuk_bin"] = df["UrineLeuk_bin"].map({">0": 1, "=0": 0})
    df["Channel_size"] = df["Channel_size"].map({"18F": 1, "20F": 0})
    df["degree_of_hydronephrosis"] = df["degree_of_hydronephrosis"].map({
        "None": 0, "Mild": 1, "Moderate": 2, "Severe": 3
    })
    df["MayoScore_bin"] = df["MayoScore_bin"].map({"≥3": 1, "<3": 0})

    # 预测
    proba = model.predict_proba(df)[0][1] * 100
    if proba < 25:
        level, color = "Low Risk", "green"
    elif proba < 50:
        level, color = "Moderate-Low Risk", "lightgreen"
    elif proba < 75:
        level, color = "Moderate-High Risk", "orange"
    else:
        level, color = "High Risk", "red"

    # 显示结果
    st.markdown("## Prediction Results")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"""
        <div style="padding:20px;border-radius:10px;background-color:{color};text-align:center;">
            <h2 style="color:white;">Risk Level: {level}</h2>
            <h3 style="color:white;">Probability: {proba:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        ### Interpretation
        - Predicted fever probability: **{proba:.2f}%**  
        - Risk level: **{level}**  
        """)
    with c2:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie([proba, 100 - proba], labels=["Fever", "No Fever"], autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

    # ——— 可选：内置你的 SHAP “手动绘图” 逻辑，保持不变 ———
    # （此处略，直接复用你原来的 try…except 代码块）

# ——— Footer ———
st.markdown("""
<div class="footer">
    © 2025 PCNL Fever Prediction Model | For academic use only.
</div>
""", unsafe_allow_html=True)
