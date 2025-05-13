# -*- coding: utf-8 -*-
"""
Created on Tue May 13 09:13:45 2025

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
    - **Mayo Score**: Mayo Surgical Complexity Score  
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

        # ——— FIXED SHAP explanations section ———
        try:
            st.markdown("## Feature Impact Analysis")
            st.info("Red features increase fever risk; blue features decrease risk.")
            
            # Calculate predicted probability
            prediction_score = model.predict_proba(df)[0][1]  # 获取阳性类别的概率
            base_value = 0.5  # 基准值（不确定时为0.5）
            
            # Get coefficients and their contributions
            coeffs = model.coef_[0]
            feature_names = df.columns.tolist()
            feature_values = df.iloc[0].values
            
            # Calculate impacts - for logistic regression can use coefficients * values
            impacts = coeffs * feature_values
            
            # Sort by absolute impact
            sorted_idx = np.argsort(np.abs(impacts))
            sorted_features = [feature_names[i] for i in sorted_idx]
            sorted_impacts = impacts[sorted_idx]
            sorted_values = feature_values[sorted_idx]
            
            # Create force plot visualization
            fig, ax = plt.subplots(figsize=(10, 3))
            
            # Setup the plot
            ax.set_xlim(0.2, 0.9)  # 根据您的预测范围调整
            ax.set_title(f"Based on feature values, predicted possibility of fever is {proba:.2f}%", fontsize=14)
            
            # 添加概率线
            ax.axvline(x=prediction_score, color='black', linestyle='-', alpha=0.5)
            ax.text(prediction_score, 0.5, f"{prediction_score:.3f}", ha='center', va='bottom', fontsize=12)
            
            # 添加左右标签
            ax.text(0.25, 0.8, "lower risk", ha='center', fontsize=10, color='blue')
            ax.text(0.85, 0.8, "higher risk", ha='center', fontsize=10, color='red')
            
            # 创建一个水平条形来表示概率范围
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # 放置刻度
            ticks = np.arange(0.2, 1.0, 0.1)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{x:.1f}" for x in ticks])
            
            # 添加特征贡献标记
            pos_features = [(feature_names[i], impacts[i], feature_values[i]) for i in range(len(impacts)) if impacts[i] > 0]
            neg_features = [(feature_names[i], impacts[i], feature_values[i]) for i in range(len(impacts)) if impacts[i] < 0]
            
            # 显示顶部贡献特征的值
            top_pos = sorted(pos_features, key=lambda x: abs(x[1]), reverse=True)[:3]
            top_neg = sorted(neg_features, key=lambda x: abs(x[1]), reverse=True)[:3]
            
            # 在图表下方添加箭头和值标签
            y_pos = -0.2
            # 正向特征（红色）
            for i, (feat, imp, val) in enumerate(top_pos):
                if feat in ["Sex", "Diabetes_mellitus", "UrineLeuk_bin", "Channel_size", "MayoScore_bin"]:
                    # 处理分类特征
                    orig_value = input_data[feat]  # 获取编码前的原始值
                    ax.annotate(f"{feat} = {orig_value}", 
                               xy=(0.3 + i*0.15, y_pos), 
                               xytext=(0.3 + i*0.15, y_pos-0.1),
                               arrowprops=dict(arrowstyle='->', color='red'),
                               ha='center', va='top', color='red')
                else:
                    ax.annotate(f"{feat} = {val:.1f}", 
                               xy=(0.3 + i*0.15, y_pos), 
                               xytext=(0.3 + i*0.15, y_pos-0.1),
                               arrowprops=dict(arrowstyle='->', color='red'),
                               ha='center', va='top', color='red')
            
            # 负向特征（蓝色）
            for i, (feat, imp, val) in enumerate(top_neg):
                if feat in ["Sex", "Diabetes_mellitus", "UrineLeuk_bin", "Channel_size", "MayoScore_bin"]:
                    # 处理分类特征
                    orig_value = input_data[feat]  # 获取编码前的原始值
                    ax.annotate(f"{feat} = {orig_value}", 
                               xy=(0.7 + i*0.12, y_pos), 
                               xytext=(0.7 + i*0.12, y_pos-0.1),
                               arrowprops=dict(arrowstyle='<-', color='blue'),
                               ha='center', va='top', color='blue')
                else:
                    ax.annotate(f"{feat} = {val:.1f}", 
                               xy=(0.7 + i*0.12, y_pos), 
                               xytext=(0.7 + i*0.12, y_pos-0.1),
                               arrowprops=dict(arrowstyle='<-', color='blue'),
                               ha='center', va='top', color='blue')
            
            # 移除y轴刻度和标签
            ax.set_yticks([])
            ax.set_yticklabels([])
            
            # 创建红蓝渐变条
            for i in range(100):
                if i < 50:  # 红色部分（左侧）
                    ax.axvspan(0.2 + i*0.007, 0.2 + (i+1)*0.007, alpha=0.6, color=plt.cm.Reds(i/50))
                else:  # 蓝色部分（右侧）
                    ax.axvspan(0.2 + i*0.007, 0.2 + (i+1)*0.007, alpha=0.6, color=plt.cm.Blues((100-i)/50))
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # 添加特征影响的文本解释
            st.subheader("Feature Impact Explanation")
            st.markdown("""
            ### How to interpret the visualization:
            - The position of the black line shows the predicted probability of fever
            - Red features on the left push the prediction towards fever
            - Blue features on the right push the prediction away from fever
            - The top influencing features are shown with their values
            """)
            
            # 表格形式展示所有特征的贡献
            st.subheader("All Feature Contributions")
            feature_contrib = pd.DataFrame({
                'Feature': feature_names,
                'Value': feature_values,
                'Impact': impacts,
                'Direction': ['Increases fever risk' if i > 0 else 'Decreases fever risk' for i in impacts]
            }).sort_values('Impact', key=abs, ascending=False)
            
            st.dataframe(feature_contrib)
            
        except Exception as e:
            st.warning(f"Could not generate feature impact visualization: {str(e)}")
            
            # Simple fallback visualization without SHAP
            try:
                st.subheader("Feature Importance (Basic Visualization)")
                
                # Use model coefficients for feature importance
                coeffs = model.coef_[0]
                feature_names = df.columns.tolist()
                
                # Create simple bar chart
                plt.figure(figsize=(10, 6))
                colors = ['red' if c > 0 else 'blue' for c in coeffs]
                plt.barh(feature_names, np.abs(coeffs), color=colors)
                plt.xlabel('Absolute Coefficient Value')
                plt.title('Feature Impact on Fever Risk')
                plt.tight_layout()
                st.pyplot(plt)
            except:
                st.error("Could not generate feature importance visualization.")

# ——— Footer ———
st.markdown("""
<div class="footer">
    © 2025 PCNL Fever Prediction Model | For academic use only.
</div>
""", unsafe_allow_html=True)

with st.expander("How to Use"):
    st.markdown("""
    1. Enter patient parameters.  
    2. Click **Predict Fever Risk**.  
    3. Review the probability and feature-impact charts.  

    **Note**: Model trained on historical data; applicability may vary.
    """)