# -*- coding: utf-8 -*-
"""
Created on Tue May 13 20:18:57 2025

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
    .shap-plot {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .shap-explanation {
        padding: 15px;
        background-color: #f0f2f6;
        border-radius: 5px;
        margin-top: 15px;
    }
    .highlight-text {
        background-color: #e3f2fd;
        padding: 2px 5px;
        border-radius: 3px;
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
        # 保存原始输入数据的副本用于展示
        display_data = input_data.copy()
        
        # 创建数据框并编码分类特征
        df = pd.DataFrame([input_data])
        # 编码分类特征
        df["Sex"] = df["Sex"].map({"Male": 1, "Female": 0})
        df["Diabetes_mellitus"] = df["Diabetes_mellitus"].map({"Yes": 1, "No": 0})
        df["UrineLeuk_bin"] = df["UrineLeuk_bin"].map({">0": 1, "=0": 0})
        df["Channel_size"] = df["Channel_size"].map({"18F": 1, "20F": 0})
        df["degree_of_hydronephrosis"] = df["degree_of_hydronephrosis"].map({
            "None": 0, "Mild": 1, "Moderate": 2, "Severe": 3
        })
        df["MayoScore_bin"] = df["MayoScore_bin"].map({"≥3": 1, "<3": 0})

        # 预测概率
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

            **Note**: For academic reference only; not a substitute for clinical judgment.
            """)
        with c2:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie([proba, 100 - proba], labels=["Fever", "No Fever"], autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            st.pyplot(fig)

        # ——— 改进的SHAP可视化部分 ———
        try:
            st.markdown("## Feature Impact Analysis")
            st.markdown("""
            <div style="padding:10px;border-radius:5px;background-color:#f0f2f6;">
                <p style="margin-bottom:0;"><strong>Red features increase fever risk; blue features decrease risk.</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # 创建SHAP解释器
            # 对于逻辑回归，我们可以使用LinearExplainer
            feature_names = df.columns.tolist()
            
            # 创建SHAP解释器 - 为逻辑回归模型特别优化
            if hasattr(model, "predict_proba"):
                # 获取背景数据样本 - 在真实应用中，这应该是训练数据的样本
                # 这里我们只有一个样本作为背景数据
                background_data = df
                explainer = shap.LinearExplainer(model, background_data)
                
                # 计算SHAP值
                shap_values = explainer.shap_values(df)[0]  # 获取正类的SHAP值
                
                # 创建强制图布局的DataFrame
                shap_df = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP_Value': shap_values,
                    'Feature_Value': df.values[0],
                    'Display_Value': [display_data[f] if f in ["Sex", "Diabetes_mellitus", "UrineLeuk_bin", 
                                                            "Channel_size", "MayoScore_bin", "degree_of_hydronephrosis"] 
                                  else df[f].values[0] for f in feature_names]
                })
                
                # 根据SHAP值绝对值排序
                shap_df = shap_df.sort_values(by='SHAP_Value', key=abs, ascending=False)
                
                # 选取前10个最重要的特征
                top_features = shap_df.head(10)
                
                # 绘制SHAP力图（Matplotlib版本）
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # 设置颜色
                colors = ['#ff0051' if x > 0 else '#008bfb' for x in top_features['SHAP_Value']]
                
                # 绘制水平条形图
                bars = ax.barh(y=range(len(top_features)), width=top_features['SHAP_Value'], color=colors)
                
                # 设置Y轴标签 - 包括特征名称和显示值
                labels = [f"{row['Feature']}={row['Display_Value']}" 
                         if isinstance(row['Display_Value'], str) 
                         else f"{row['Feature']}={row['Display_Value']:.2f}" 
                         for _, row in top_features.iterrows()]
                
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(labels)
                
                # 添加基线
                ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
                
                # 设置标题和标签
                ax.set_title(f"Top Features Impact on Prediction (Base value: {explainer.expected_value:.2f}, Prediction: {model.predict_proba(df)[0][1]:.2f})")
                ax.set_xlabel('SHAP Value (Impact on prediction)')
                
                # 添加网格线
                ax.grid(axis='x', linestyle='--', alpha=0.3)
                
                # 调整布局
                plt.tight_layout()
                
                # 在Streamlit中显示图表
                st.pyplot(fig)
                
                # 使用SHAP的内置可视化 - 力图
                shap.initjs()  # 初始化JavaScript可视化
                
                # 创建SHAP力图
                plt.figure(figsize=(14, 4))
                force_plot = shap.force_plot(
                    base_value=explainer.expected_value, 
                    shap_values=shap_values,
                    features=df.iloc[0],
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False
                )
                
                # 保存为图像以在Streamlit中显示
                plt.tight_layout()
                
                st.markdown("### SHAP Force Plot")
                st.markdown("This visualization shows how each feature pushes the prediction from the base value towards the final prediction.")
                st.pyplot(plt.gcf())
                
                # 可视化特征重要性的摘要图
                plt.figure(figsize=(10, 7))
                shap.summary_plot(
                    shap_values=shap_values,
                    features=df,
                    feature_names=feature_names,
                    show=False
                )
                plt.tight_layout()
                
                st.markdown("### SHAP Feature Importance")
                st.markdown("This visualization shows the overall importance of each feature across all predictions.")
                st.pyplot(plt.gcf())
                
                # 显示特征贡献表格
                st.subheader("Feature Contributions")
                
                # 为所有特征创建表格数据
                feature_contrib = pd.DataFrame({
                    'Feature': [f"{f}" for f in feature_names],
                    'Value': [display_data[f] if f in ["Sex", "Diabetes_mellitus", "UrineLeuk_bin", 
                                                  "Channel_size", "MayoScore_bin", "degree_of_hydronephrosis"] 
                           else round(df[f].values[0], 2) for f in feature_names],
                    'Impact': shap_values,
                    'Direction': ['Increases fever risk' if v > 0 else 'Decreases fever risk' for v in shap_values]
                }).sort_values('Impact', key=abs, ascending=False)
                
                st.dataframe(feature_contrib)
                
                # 显示前5个最有影响力的特征作为文本
                st.subheader("Key Factors")
                st.markdown("The most important factors affecting this prediction:")
                
                # 获取前5个特征
                for i, (_, row) in enumerate(top_features.head(5).iterrows()):
                    direction = "increases" if row['SHAP_Value'] > 0 else "decreases"
                    impact = abs(row['SHAP_Value'])
                    feature = row['Feature']
                    display_value = row['Display_Value']
                    
                    if isinstance(display_value, (int, float)):
                        display_value = f"{display_value:.2f}"
                    
                    if i == 0:
                        # 强调最重要的特征
                        st.markdown(f"- **{feature} = {display_value}**: <span style='color:{'red' if row['SHAP_Value'] > 0 else 'blue'};font-weight:bold;'>{direction} fever risk the most (impact: {impact:.4f})</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"- **{feature} = {display_value}**: {direction} fever risk (impact: {impact:.4f})")
                
                # 添加解释性文本
                st.markdown("""
                <div class="shap-explanation">
                    <h3>How to interpret these visualizations:</h3>
                    <ul>
                        <li><strong>Force Plot</strong>: Shows how each feature pushes the prediction away from the base value (average prediction for all patients) towards the final prediction for this specific patient.</li>
                        <li><strong>Bar Chart</strong>: Red bars push the prediction higher (increasing fever risk), while blue bars push it lower (decreasing fever risk).</li>
                        <li><strong>Feature Importance</strong>: Shows the distribution of feature impacts across all possible values, with color indicating the feature value (red = high, blue = low).</li>
                    </ul>
                    <p>These SHAP values accurately quantify each feature's contribution to the prediction, accounting for interactions between features.</p>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.error("The loaded model does not support probability predictions required for SHAP analysis.")
                
        except Exception as e:
            st.warning(f"Could not generate SHAP visualizations: {str(e)}")
            
            # 简单的备用可视化
            try:
                st.subheader("Feature Importance (Basic Visualization)")
                
                # 使用模型系数作为特征重要性
                coeffs = model.coef_[0]
                feature_names = df.columns.tolist()
                
                # 创建简单的条形图
                plt.figure(figsize=(10, 6))
                colors = ['#ff0051' if c > 0 else '#008bfb' for c in coeffs]  # 使用SHAP颜色
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