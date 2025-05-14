# -*- coding: utf-8 -*-
"""
Created on Wed May 14 11:41:44 2025

@author: LENOVO
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import joblib
import shap
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import os

# 设置页面配置
st.set_page_config(
    page_title="PCNL Post-Operative Fever Prediction Model",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加CSS样式
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
    }
    h1 {
        color: #2C3E50;
    }
    h2 {
        color: #3498DB;
    }
    .footer {
        margin-top: 2rem;
        text-align: center;
        color: #7F8C8D;
    }
</style>
""", unsafe_allow_html=True)

# 显示标题与说明
st.title("PCNL Post-Operative Fever Prediction Model")
st.markdown("### A machine learning-based tool to predict post-operative fever risk after percutaneous nephrolithotomy")

# 创建侧边栏信息
with st.sidebar:
    st.header("About this Model")
    st.info(
        """
        This prediction model is based on clinical features to assess the risk of fever after 
        Percutaneous Nephrolithotomy (PCNL).
        
        Please enter the patient's information on the right to obtain the prediction results.
        """
    )
    st.header("Feature Description")
    st.markdown("""
    - **LMR**: Lymphocyte to Monocyte Ratio
    - **PLR**: Platelet to Lymphocyte Ratio
    - **BMI**: Body Mass Index
    - **Mayo Score**: Mayo Surgical Complexity Score
    """)

# 确保模型文件存在
def load_model():
    try:
        # 尝试加载模型
        return joblib.load('LR.pkl')
    except:
        # 如果加载失败，显示错误信息
        st.error("Model file 'LR.pkl' not found. Please ensure the model file is uploaded to the same directory as the application.")
        return None

# 定义特征范围
feature_ranges = {
    # 原始经皮肾镜术后发热预测模型特征
    "LMR": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 5.0, 
            "description": "Lymphocyte to Monocyte Ratio"},
    "Preoperative_N": {"type": "numerical", "min": 0.0, "max": 30.0, "default": 4.0,
                        "description": "Preoperative Neutrophil Count (×10^9/L)"},
    "Operative_time": {"type": "numerical", "min": 10, "max": 300, "default": 60,
                      "description": "Operative Time (minutes)"},
    "Preoperative_WBC": {"type": "numerical", "min": 0.0, "max": 30.0, "default": 7.0,
                         "description": "Preoperative White Blood Cell Count (×10^9/L)"},
    "Preoperative_L": {"type": "numerical", "min": 0.0, "max": 10.0, "default": 1.8,
                      "description": "Preoperative Lymphocyte Count (×10^9/L)"},
    "PLR": {"type": "numerical", "min": 0.0, "max": 1000.0, "default": 120.0,
           "description": "Platelet to Lymphocyte Ratio"},
    "Preoperative_hemoglobin": {"type": "numerical", "min": 50.0, "max": 200.0, "default": 130.0,
                               "description": "Preoperative Hemoglobin (g/L)"},
    "Number_of_stones": {"type": "numerical", "min": 1, "max": 20, "default": 1,
                        "description": "Number of Stones"},
    "BMI": {"type": "numerical", "min": 10.0, "max": 50.0, "default": 24.0,
           "description": "Body Mass Index (kg/m²)"},
    
    "Sex": {"type": "categorical", "options": ["Male", "Female"], "default": "Male",
            "description": "Sex"},
    "Diabetes_mellitus": {"type": "categorical", "options": ["No", "Yes"], "default": "No",
                          "description": "Diabetes Mellitus"},
    "UrineLeuk_bin": {"type": "categorical", "options": ["=0", ">0"], "default": "=0",
                      "description": "Urine Leukocytes"},
    "Channel_size": {"type": "categorical", "options": ["18F", "20F"], "default": "18F",
                     "description": "Channel Size"},
    "degree_of_hydronephrosis": {"type": "categorical", "options": ["None", "Mild", "Moderate", "Severe"], "default": "None",
                                 "description": "Degree of Hydronephrosis"},
    "MayoScore_bin": {"type": "categorical", "options": ["<3", "≥3"], "default": "<3",
                      "description": "Mayo Score"}
}

# 创建用户输入页面布局
st.header("Enter Patient Information")

# 使用列布局改善用户界面
col1, col2, col3 = st.columns(3)

# 创建空字典存储特征值
input_features = {}

# 将特征分配到列中
feature_columns = {
    0: col1,
    1: col2,
    2: col3
}

# 将特征分组到列中
i = 0
for feature, properties in feature_ranges.items():
    col = feature_columns[i % 3]
    with col:
        if properties["type"] == "numerical":
            input_features[feature] = st.number_input(
                label=f"{properties['description']} ({feature})",
                min_value=float(properties["min"]),
                max_value=float(properties["max"]),
                value=float(properties["default"]),
                help=f"Range: {properties['min']} - {properties['max']}"
            )
        elif properties["type"] == "categorical":
            input_features[feature] = st.selectbox(
                label=f"{properties['description']} ({feature})",
                options=properties["options"],
                index=properties["options"].index(properties["default"]),
                help=f"Select an option"
            )
    i += 1

# 添加分隔线
st.markdown("---")

# 添加预测按钮
predict_button = st.button("Predict Fever Risk", use_container_width=True)

# 当按钮被点击时进行预测
if predict_button:
    # 加载模型
    model = load_model()
    
    if model:
        # 获取分类特征和数值特征列表
        categorical_features = [f for f, p in feature_ranges.items() if p["type"] == "categorical"]
        numerical_features = [f for f, p in feature_ranges.items() if p["type"] == "numerical"]
        
        # 准备数据框
        input_df = pd.DataFrame([input_features])
        
        # 处理分类特征 - 根据模型训练方式调整
        for feature in categorical_features:
            # 假设模型训练时使用了这些编码方式
            if feature == "Sex":
                input_df[feature] = 1 if input_features[feature] == "Male" else 0
            elif feature == "Diabetes_mellitus":
                input_df[feature] = 1 if input_features[feature] == "Yes" else 0
            elif feature == "UrineLeuk_bin":
                input_df[feature] = 1 if input_features[feature] == ">0" else 0
            elif feature == "Channel_size":
                input_df[feature] = 1 if input_features[feature] == "18F" else 0
            elif feature == "degree_of_hydronephrosis":
                # 假设编码: None=0, Mild=1, Moderate=2, Severe=3
                hydro_map = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
                input_df[feature] = hydro_map[input_features[feature]]
            elif feature == "MayoScore_bin":
                input_df[feature] = 1 if input_features[feature] == "≥3" else 0
        
        try:
            # 进行预测
            predicted_proba = model.predict_proba(input_df)[0]
            # 假设模型是二分类，1表示发热
            fever_probability = predicted_proba[1] * 100
            
            # 显示结果
            st.markdown("## Prediction Results")
            
            # 创建结果显示区域
            result_col1, result_col2 = st.columns([2, 1])
            
            with result_col1:
                # 根据概率值显示不同的风险级别
                if fever_probability < 25:
                    risk_level = "Low Risk"
                    color = "green"
                elif fever_probability < 50:
                    risk_level = "Moderate-Low Risk"
                    color = "lightgreen"
                elif fever_probability < 75:
                    risk_level = "Moderate-High Risk"
                    color = "orange"
                else:
                    risk_level = "High Risk"
                    color = "red"
                
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color: {color}; text-align: center;">
                    <h2 style="color: white;">Post-operative Fever Risk: {risk_level}</h2>
                    <h3 style="color: white;">Predicted Probability: {fever_probability:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # 添加结果解释
                st.markdown(f"""
                ### Result Interpretation
                - The predicted probability of post-operative fever for this patient is **{fever_probability:.2f}%**
                - Risk Level: **{risk_level}**
                
                **Note**: This prediction is for clinical reference only and should not replace professional medical judgment.
                """)
            
            with result_col2:
                # 创建简单的概率可视化
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie([fever_probability, 100-fever_probability], 
                       labels=["Fever Risk", "No Fever Risk"],
                       colors=[color, "lightgrey"],
                       autopct='%1.1f%%',
                       startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
            
            # 尝试使用SHAP进行解释 - 修复SHAP相关代码
            try:
                st.markdown("## Feature Impact Analysis")
                st.info("The chart below shows how each feature influences the prediction. Features pushing to the right (red) increase fever risk, while those pushing to the left (blue) decrease risk.")
                
                # 创建合适的背景数据
                # 为了避免"index 1 is out of bounds for axis 0 with size 1"错误，
                # 我们需要确保正确处理二分类预测问题
                
                # 方法1：使用KernelExplainer，确保正确指定输出索引
                # 创建简单的预测函数，只返回正类的概率
                def predict_proba_positive_class(X):
                    return model.predict_proba(X)[:, 1]
                
                # 创建背景数据 - 这里使用输入数据本身作为背景
                # 或者创建一个更大的样本集合
                background_data = pd.concat([input_df] * 10, ignore_index=True)
                
                # 创建SHAP解释器
                explainer = shap.KernelExplainer(predict_proba_positive_class, background_data)
                
                # 计算SHAP值 - 注意这里不会有索引问题，因为函数只返回一个类别的概率
                shap_values = explainer.shap_values(input_df)
                
                # 创建SHAP力量图
                st.subheader("SHAP Force Plot")
                fig_force = plt.figure(figsize=(12, 4))
                shap.force_plot(
                    explainer.expected_value, 
                    shap_values[0], 
                    input_df.iloc[0], 
                    feature_names=input_df.columns.tolist(),
                    matplotlib=True,
                    show=False
                )
                plt.tight_layout()
                st.pyplot(fig_force)
                plt.clf()
                
                # 创建SHAP瀑布图
                st.subheader("SHAP Waterfall Plot")
                fig_waterfall = plt.figure(figsize=(10, 8))
                
                # 使用更安全的方法创建瀑布图
                shap.plots._waterfall.waterfall_legacy(
                    explainer.expected_value, 
                    shap_values[0], 
                    feature_names=input_df.columns.tolist(),
                    show=False
                )
                plt.tight_layout()
                st.pyplot(fig_waterfall)
                
                # 显示特征重要性表格
                st.subheader("Feature Importance")
                feature_importance = pd.DataFrame({
                    'Feature': input_df.columns.tolist(),
                    'SHAP Value': np.abs(shap_values[0])
                }).sort_values('SHAP Value', ascending=False)
                
                st.table(feature_importance)
                
            except Exception as e:
                st.warning(f"Unable to generate SHAP explanation: {str(e)}")
                st.markdown("""
                #### Alternative Feature Impact Visualization
                
                Since SHAP visualization is not available, we'll show a simplified feature impact analysis based on the model's coefficients.
                """)
                
                # 如果SHAP不可用，创建简单的基于模型系数的特征影响可视化
                try:
                    if hasattr(model, 'coef_'):
                        coef_df = pd.DataFrame({
                            'Feature': input_df.columns.tolist(),
                            'Coefficient': model.coef_[0]
                        }).sort_values('Coefficient', ascending=False)
                        
                        # 创建系数图
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = ['green' if c < 0 else 'red' for c in coef_df['Coefficient']]
                        ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
                        ax.set_xlabel('Coefficient Value')
                        ax.set_ylabel('Feature')
                        ax.set_title('Feature Coefficients - Red increases fever risk, Green decreases')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # 显示特征系数表格
                        st.subheader("Feature Coefficients")
                        st.table(coef_df)
                    else:
                        st.info("Feature impact analysis is not available for this model type.")
                except Exception as e:
                    st.error(f"Unable to generate alternative feature impact visualization: {str(e)}")
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.markdown("""
            Possible reasons:
            1. Input data format does not match model expectations
            2. Model file may be corrupted or incompatible
            """)

# 添加页脚
st.markdown("""
<div class="footer">
    <p>© 2025 PCNL Post-Operative Fever Prediction Model | This tool is for clinical reference only and should not replace professional medical judgment</p>
</div>
""", unsafe_allow_html=True)

# 添加"如何使用"折叠面板
with st.expander("How to Use This Tool"):
    st.markdown("""
    1. Enter the patient's clinical parameters in the form above
    2. Click the "Predict Fever Risk" button
    3. Review the prediction results and feature impact analysis
    4. Use the results as a reference for clinical decision-making
    
    **Notes**:
    - All values must be within the specified ranges
    - For missing data, it's recommended to use common clinical default values
    - This model is trained on historical data and may not apply to all clinical scenarios
    """)