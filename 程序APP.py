# -*- coding: utf-8 -*-
"""
Created on Wed May 14 08:53:23 2025

@author: LENOVO
"""

# -*- coding: utf-8 -*-
"""
PCNL Post-Operative Fever Prediction Web App
Author: LENOVO
"""

# 必须最先设置页面配置
import streamlit as st
st.set_page_config(
    page_title="PCNL Post-Operative Fever Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 后续导入模块
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.linear_model import LogisticRegression


@st.cache_resource
def load_model():
    try:
        return joblib.load("LR.pkl")
    except FileNotFoundError:
        st.error("模型文件 'LR.pkl' 未找到，请将其与该脚本放在同一目录下。")
        return None


def main():
    st.title("PCNL Post-Operative Fever Prediction")
    st.markdown("### A machine learning–based tool to estimate fever risk after PCNL")

    with st.sidebar:
        st.header("About the Model")
        st.info(
            "This logistic regression model is trained on historical clinical data "
            "to predict the risk of post-operative fever after percutaneous nephrolithotomy (PCNL)."
        )
        st.header("Feature Descriptions")
        st.markdown("""
        - **LMR**: Lymphocyte-to-Monocyte Ratio  
        - **PLR**: Platelet-to-Lymphocyte Ratio  
        - **BMI**: Body Mass Index  
        - **Mayo Score**: Mayo Surgical Complexity Score  
        """)

    # 预测特征设置
    feature_ranges = {
        "LMR": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 5.0},
        "Preoperative_N": {"type": "numerical", "min": 0.0, "max": 30.0, "default": 4.0},
        "Operative_time": {"type": "numerical", "min": 10, "max": 300, "default": 60},
        "Preoperative_WBC": {"type": "numerical", "min": 0.0, "max": 30.0, "default": 7.0},
        "Preoperative_L": {"type": "numerical", "min": 0.0, "max": 10.0, "default": 1.8},
        "PLR": {"type": "numerical", "min": 0.0, "max": 1000.0, "default": 120.0},
        "Preoperative_hemoglobin": {"type": "numerical", "min": 50.0, "max": 200.0, "default": 130.0},
        "Number_of_stones": {"type": "numerical", "min": 1, "max": 20, "default": 1},
        "BMI": {"type": "numerical", "min": 10.0, "max": 50.0, "default": 24.0},
        "Sex": {"type": "categorical", "options": ["Male", "Female"], "default": "Male"},
        "Diabetes_mellitus": {"type": "categorical", "options": ["No", "Yes"], "default": "No"},
        "UrineLeuk_bin": {"type": "categorical", "options": ["=0", ">0"], "default": "=0"},
        "Channel_size": {"type": "categorical", "options": ["18F", "20F"], "default": "18F"},
        "degree_of_hydronephrosis": {"type": "categorical", "options": ["None", "Mild", "Moderate", "Severe"], "default": "None"},
        "MayoScore_bin": {"type": "categorical", "options": ["<3", "≥3"], "default": "<3"}
    }

    # 输入界面
    st.header("Enter Patient Parameters")
    cols = st.columns(3)
    input_data = {}
    for idx, (feat, cfg) in enumerate(feature_ranges.items()):
        col = cols[idx % 3]
        with col:
            if cfg["type"] == "numerical":
                input_data[feat] = st.number_input(
                    label=feat,
                    min_value=cfg["min"],
                    max_value=cfg["max"],
                    value=cfg["default"]
                )
            else:
                input_data[feat] = st.selectbox(
                    label=feat,
                    options=cfg["options"],
                    index=cfg["options"].index(cfg["default"])
                )

    st.markdown("---")
    if st.button("Predict Fever Risk", use_container_width=True):
        model = load_model()
        if model:
            df = pd.DataFrame([input_data])
            df["Sex"] = df["Sex"].map({"Male": 1, "Female": 0})
            df["Diabetes_mellitus"] = df["Diabetes_mellitus"].map({"Yes": 1, "No": 0})
            df["UrineLeuk_bin"] = df["UrineLeuk_bin"].map({">0": 1, "=0": 0})
            df["Channel_size"] = df["Channel_size"].map({"18F": 1, "20F": 0})
            df["degree_of_hydronephrosis"] = df["degree_of_hydronephrosis"].map({"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3})
            df["MayoScore_bin"] = df["MayoScore_bin"].map({"≥3": 1, "<3": 0})

            proba = model.predict_proba(df)[0][1] * 100
            level = "Low Risk" if proba < 25 else "Moderate-Low Risk" if proba < 50 else "Moderate-High Risk" if proba < 75 else "High Risk"
            color = "green" if proba < 25 else "lightgreen" if proba < 50 else "orange" if proba < 75 else "red"

            st.markdown("## Prediction Results")
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"""
                <div style="padding:20px;border-radius:10px;background-color:{color};text-align:center;">
                    <h2 style="color:white;">Risk Level: {level}</h2>
                    <h3 style="color:white;">Probability: {proba:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie([proba, 100 - proba], labels=["Fever", "No Fever"], autopct="%1.1f%%", startangle=90)
                ax.axis("equal")
                st.pyplot(fig)

            # SHAP 力图
            try:
                st.subheader("SHAP Force Plot (Simplified)")
                explainer = shap.LinearExplainer(model, df, feature_perturbation="interventional")
                shap_values = explainer.shap_values(df)

                fig = plt.figure(figsize=(12, 2))
                shap.force_plot(
                    base_value=explainer.expected_value,
                    shap_values=shap_values[0],
                    features=df.iloc[0],
                    matplotlib=True,
                    show=False
                )
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"SHAP force plot could not be rendered: {str(e)}")

    st.markdown("""
    <div class="footer">
        © 2025 PCNL Fever Prediction Model | For academic use only.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("How to Use"):
        st.markdown("""
        1. Enter patient parameters.  
        2. Click **Predict Fever Risk**.  
        3. Review the prediction results and SHAP feature impact chart.  

        **Note**: Model trained on historical data; clinical use requires validation.
        """)


if __name__ == "__main__":
    main()
