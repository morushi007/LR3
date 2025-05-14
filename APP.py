# -*- coding: utf-8 -*-
"""
Created on Wed May 14 10:03:18 2025

@author: LENOVO
"""
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 13:48:35 2025

@author: LENOVO
"""

# app.py
# -*- coding: utf-8 -*-
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
                options=cfg["options"],
                index=cfg["options"].index(cfg["default"])
            )

st.markdown("---")
if st.button("Predict Fever Risk", use_container_width=True):
    model = load_model()
    if model:
        df = pd.DataFrame([input_data])
        # Encode categorical features
        df["Sex"] = df["Sex"].map({"Male": 1, "Female": 0})
        df["Diabetes_mellitus"] = df["Diabetes_mellitus"].map({"Yes": 1, "No": 0})
        df["UrineLeuk_bin"] = df["UrineLeuk_bin"].map({">0": 1, "=0": 0})
        df["Channel_size"] = df["Channel_size"].map({"18F": 1, "20F": 0})
        df["degree_of_hydronephrosis"] = df["degree_of_hydronephrosis"].map({
            "None": 0, "Mild": 1, "Moderate": 2, "Severe": 3
        })
        df["MayoScore_bin"] = df["MayoScore_bin"].map({"≥3": 1, "<3": 0})

        # Predict probability
        proba = model.predict_proba(df)[0][1] * 100
        if proba < 25:
            level, color = "Low Risk", "green"
        elif proba < 50:
            level, color = "Moderate-Low Risk", "lightgreen"
        elif proba < 75:
            level, color = "Moderate-High Risk", "orange"
        else:
            level, color = "High Risk", "red"

        # Display results
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

            **Note**: For academic reference only; not a substitute for clinical judgment.
            """)
        with c2:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie([proba, 100 - proba], labels=["Fever", "No Fever"], autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            st.pyplot(fig)

        # --- SHAP Force Plot visualization with proper SHAP library ---
        try:
            st.markdown("## Feature Impact Analysis")
            st.markdown("""
            <div style="padding:10px;border-radius:5px;background-color:#f0f2f6;">
                <p style="margin-bottom:0;"><strong>红色特征增加发热风险；蓝色特征降低发热风险。</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create SHAP explainer
            explainer = shap.LinearExplainer(model, df)
            shap_values = explainer.shap_values(df)
            
            # Force plot for the single prediction
            st.subheader("SHAP Force Plot")
            fig, ax = plt.subplots(figsize=(12, 5))
            
            # Create the force plot using SHAP directly
            shap.force_plot(
                explainer.expected_value, 
                shap_values[0], 
                df.iloc[0],
                matplotlib=True,
                show=False,
                figsize=(12, 5),
                text_rotation=45,
                contribution_threshold=0.05  # Only show significant features
            )
            plt.tight_layout()
            st.pyplot(fig)
            
            # Add waterfall plot for more detailed visualization
            st.subheader("特征贡献瀑布图")
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value,
                shap_values[0],
                feature_names=df.columns.tolist(),
                max_display=10,
                show=False
            )
            plt.tight_layout()
            st.pyplot(fig)
            
            # Explanation
            st.subheader("如何解读这些可视化")
            st.markdown("""
            - **力图 (Force Plot)**: 展示每个特征如何将预测从基准值（模型输出平均值）推向最终预测。
                - **红色特征** 增加预测值（提高发热风险）
                - **蓝色特征** 降低预测值（降低发热风险）
                - 每个彩色段的宽度表示该特征影响的大小
            
            - **瀑布图 (Waterfall Plot)**: 展示特征对预测的累积效果。
                - 从基准值开始，每个特征要么增加要么减少预测值
                - 特征按其影响大小排序
            """)
            
            # Display feature contributions table
            st.subheader("所有特征贡献")
            
            # Get original feature values (before encoding)
            orig_values = []
            for feat in df.columns:
                if feat in ["Sex", "Diabetes_mellitus", "UrineLeuk_bin", "Channel_size", "MayoScore_bin", "degree_of_hydronephrosis"]:
                    orig_values.append(input_data[feat])
                else:
                    orig_values.append(df[feat].iloc[0])
            
            # Create DataFrame for display
            feature_contrib = pd.DataFrame({
                '特征': df.columns,
                '输入值': orig_values,
                '影响': shap_values[0],
                '方向': ['增加发热风险' if sv > 0 else '降低发热风险' for sv in shap_values[0]]
            }).sort_values('影响', key=abs, ascending=False)
            
            st.dataframe(feature_contrib)
            
            # Display the top 5 influential features as text
            st.subheader("主要特征影响")
            st.markdown("影响预测的关键因素:")
            
            # Sort features by absolute impact
            sorted_features = sorted(zip(df.columns, shap_values[0], orig_values), 
                                    key=lambda x: abs(x[1]), 
                                    reverse=True)
            
            for feat, impact, val in sorted_features[:5]:
                direction = "增加" if impact > 0 else "降低"
                if feat in ["Sex", "Diabetes_mellitus", "UrineLeuk_bin", "Channel_size", "MayoScore_bin", "degree_of_hydronephrosis"]:
                    st.markdown(f"- **{feat} = {val}**: {direction}发热风险")
                else:
                    st.markdown(f"- **{feat} = {val:.2f}**: {direction}发热风险")
            
        except Exception as e:
            st.warning(f"无法生成SHAP可视化: {str(e)}")
            
            # Fallback to simple feature importance visualization
            st.subheader("特征重要性 (基础可视化)")
            
            # Use model coefficients and input values for feature importance
            coeffs = model.coef_[0]
            feature_names = df.columns.tolist()
            feature_values = df.iloc[0].values
            
            # Calculate impact as coefficient * value
            impacts = coeffs * feature_values
            
            # Create bar chart of impacts
            plt.figure(figsize=(10, 6))
            y_pos = np.arange(len(feature_names))
            colors = ['#ff0051' if i > 0 else '#008bfb' for i in impacts]
            
            # Sort by absolute impact
            sorted_indices = np.argsort(np.abs(impacts))[::-1]
            sorted_impacts = impacts[sorted_indices]
            sorted_names = [feature_names[i] for i in sorted_indices]
            sorted_colors = [colors[i] for i in sorted_indices]
            
            plt.barh(y_pos, sorted_impacts, color=sorted_colors)
            plt.yticks(y_pos, sorted_names)
            plt.xlabel('对预测的影响')
            plt.title('特征对发热风险的影响')
            plt.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
            plt.tight_layout()
            st.pyplot(plt)

# ——— Footer ———
st.markdown("""
<div class="footer">
    © 2025 PCNL 发热预测模型 | 仅供学术使用。
</div>
""", unsafe_allow_html=True)

with st.expander("使用说明"):
    st.markdown("""
    1. 输入患者参数。
    2. 点击**预测发热风险**。
    3. 查看概率和特征影响图表。

    **注意**：模型基于历史数据训练；适用性可能因个体而异。
    """)
