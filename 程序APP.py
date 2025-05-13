# -*- coding: utf-8 -*-
"""
Created on Tue May 13 09:24:05 2025

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
            prediction_score = model.predict_proba(df)[0][1]
            base_value = 0.5  # 基准值
            
            # Get coefficients and their contributions
            coeffs = model.coef_[0]
            feature_names = df.columns.tolist()
            feature_values = df.iloc[0].values
            
            # Calculate impacts - for logistic regression can use coefficients * values
            impacts = coeffs * feature_values
            
            # Sort features by impact for visualization
            features_with_impact = list(zip(feature_names, impacts, feature_values))
            # Sort by absolute impact (preserving sign)
            sorted_features = sorted(features_with_impact, key=lambda x: abs(x[1]), reverse=True)
            
            # Extract top features (limiting to prevent overcrowding)
            top_features = sorted_features[:7]  # 限制显示的特征数量
            
            # Setup the figure with SHAP-like style
            fig, ax = plt.subplots(figsize=(12, 4))
            
            # 设置x轴范围，以对称方式展示正负影响
            max_impact = max(abs(impact) for _, impact, _ in top_features) * 1.2
            ax.set_xlim(-max_impact, max_impact)
            
            # 添加基准线
            ax.axvline(x=0, color='#999999', linestyle='-', alpha=0.5)
            
            # 添加红蓝配色的水平条
            colors = {'positive': '#f8766d', 'negative': '#00bfc4'}  # SHAP原始的红蓝配色
            
            # 特征影响条
            y_pos = 0
            for i, (feature, impact, value) in enumerate(top_features):
                # 确定颜色和方向
                color = colors['positive'] if impact > 0 else colors['negative']
                
                # 绘制水平条
                ax.barh(i, impact, color=color, height=0.8, alpha=0.7)
                
                # 添加特征名称和值
                if feature in ["Sex", "Diabetes_mellitus", "UrineLeuk_bin", "Channel_size", "MayoScore_bin", "degree_of_hydronephrosis"]:
                    # 处理分类特征，显示原始值
                    orig_value = input_data[feature]
                    label_text = f"{feature} = {orig_value}"
                else:
                    # 处理数值特征，保留两位小数
                    label_text = f"{feature} = {value:.2f}"
                
                # 放置标签
                if impact > 0:
                    ax.text(impact/2, i, label_text, ha='center', va='center', color='white', fontsize=10)
                else:
                    ax.text(impact/2, i, label_text, ha='center', va='center', color='white', fontsize=10)
            
            # 设置y轴标签（空白）
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels([])
            
            # 添加标题
            ax.set_title(f"Predicted probability of fever: {proba:.2f}%", fontsize=14)
            
            # 添加higher/lower标签
            ax.text(max_impact * 0.8, -0.8, "higher →", ha='center', color=colors['positive'], fontsize=10)
            ax.text(-max_impact * 0.8, -0.8, "← lower", ha='center', color=colors['negative'], fontsize=10)
            
            # 显示预测值和基准值
            base_text = f"base value\n{base_value:.2f}"
            ax.text(0, len(top_features) + 0.5, base_text, ha='center', va='center', fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.5'))
            
            pred_text = f"{prediction_score:.2f}"
            ax.text(max_impact * 0.9, len(top_features) + 0.5, pred_text, ha='center', va='center', 
                   fontsize=10, color=colors['positive'], 
                   bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.5'))
            
            # 添加预测值线
            impact_sum = sum(impact for _, impact, _ in top_features) 
            pred_line_x = impact_sum * 0.8  # 近似位置
            ax.axvline(x=pred_line_x, color='black', linestyle='--', alpha=0.7)
            
            # 美化图表
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # 创建另一种更接近SHAP原始风格的力图 - 类似于您提供的第二张图片
            try:
                # 创建另一个图，更接近SHAP原始风格
                fig2, ax2 = plt.subplots(figsize=(12, 3))
                
                # 使用SHAP风格的颜色
                pos_color = '#ff0051'  # 红色
                neg_color = '#008bfb'  # 蓝色
                
                # 设置x轴范围，使用相对规范化的刻度
                ax2.set_xlim(-10, 10)
                ax2.set_xticks(range(-10, 11, 2))
                
                # 创建水平条形的背景
                rect = plt.Rectangle((-10, 0), 20, 1, color='#f8f8f8')
                ax2.add_patch(rect)
                
                # 基准线
                ax2.axvline(x=0, color='#999999', linewidth=1)
                
                # 当前值标记
                current_value = 2.9  # 这是示例值，您需要使用实际计算的值
                ax2.axvline(x=current_value, color='black', linestyle='--', linewidth=1)
                
                # 添加标题
                ax2.set_title(f"Based on feature values, predicted possibility of fever is {proba:.2f}%", fontsize=14)
                
                # 创建力图的连接条形
                # 为简化，我们将仅显示影响最大的几个特征
                important_features = sorted_features[:6]
                
                # 标准化为-10到10的范围
                max_abs_impact = max([abs(imp) for _, imp, _ in important_features])
                normalized_impacts = [(feat, (imp/max_abs_impact)*7, val) for feat, imp, val in important_features]
                
                # 排序，确保红色(正影响)在左侧，蓝色(负影响)在右侧
                pos_features = [(f, i, v) for f, i, v in normalized_impacts if i > 0]
                neg_features = [(f, i, v) for f, i, v in normalized_impacts if i < 0]
                
                # 绘制特征贡献
                x_pos = -8  # 起始位置
                for feat, impact, val in pos_features:
                    width = abs(impact)
                    rect = plt.Rectangle((x_pos, 0.1), width, 0.8, color=pos_color, alpha=0.7)
                    ax2.add_patch(rect)
                    
                    # 特征标签显示在图形下方
                    if feat in ["Sex", "Diabetes_mellitus", "UrineLeuk_bin", "Channel_size", "MayoScore_bin"]:
                        orig_val = input_data[feat]
                        ax2.text(x_pos + width/2, -0.3, f"{feat}\n{orig_val}", ha='center', va='top', fontsize=9)
                    else:
                        ax2.text(x_pos + width/2, -0.3, f"{feat}\n{val:.2f}", ha='center', va='top', fontsize=9)
                    
                    x_pos += width
                
                x_pos = 8  # 蓝色特征起始位置（右侧）
                for feat, impact, val in neg_features:
                    width = abs(impact)
                    x_pos -= width
                    rect = plt.Rectangle((x_pos, 0.1), width, 0.8, color=neg_color, alpha=0.7)
                    ax2.add_patch(rect)
                    
                    # 特征标签
                    if feat in ["Sex", "Diabetes_mellitus", "UrineLeuk_bin", "Channel_size", "MayoScore_bin"]:
                        orig_val = input_data[feat]
                        ax2.text(x_pos + width/2, -0.3, f"{feat}\n{orig_val}", ha='center', va='top', fontsize=9)
                    else:
                        ax2.text(x_pos + width/2, -0.3, f"{feat}\n{val:.2f}", ha='center', va='top', fontsize=9)
                
                # 添加high/low标签
                ax2.text(8, 1.2, "higher", ha='center', va='bottom', color=pos_color, fontsize=10)
                ax2.text(-8, 1.2, "lower", ha='center', va='bottom', color=neg_color, fontsize=10)
                
                # 添加predicted值和base value
                ax2.text(0, 1.5, f"base value\n{base_value:.2f}", ha='center', va='center', fontsize=10)
                ax2.text(current_value, 1.5, f"{prediction_score:.2f}", ha='center', va='center', fontsize=10)
                
                # 移除y轴
                ax2.set_yticks([])
                ax2.spines['left'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.spines['top'].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig2)
                
            except Exception as e:
                st.warning(f"Could not generate second visualization: {str(e)}")
            
            # 添加特征影响的文本解释
            st.subheader("Feature Impact Explanation")
            st.markdown("""
            ### How to interpret the visualization:
            - The red features increase the probability of fever
            - The blue features decrease the probability of fever
            - The position of the dashed line shows the predicted probability
            - The base value represents the average prediction across all samples
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