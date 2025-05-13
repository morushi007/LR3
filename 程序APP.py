# -*- coding: utf-8 -*-
"""
Created on Tue May 13 20:25:29 2025

@author: LENOVO
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 13 13:48:35 2025

@author: LENOVO
"""

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression

# ——— Page configuration - 必须是第一个Streamlit命令 ———
st.set_page_config(
    page_title="PCNL Fever Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ——— 然后是自定义CSS ———
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

        # ——— 只保留特征影响力图 ———
        try:
            st.markdown("## Feature Impact Analysis")
            st.markdown("""
            <div style="padding:10px;border-radius:5px;background-color:#f0f2f6;">
                <p style="margin-bottom:0;"><strong>Red bars increase fever risk; blue bars decrease risk.</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # 计算特征影响
            coeffs = model.coef_[0]
            feature_names = df.columns.tolist()
            feature_values = df.iloc[0].values
            impacts = coeffs * feature_values
            
            # 创建力图
            fig, ax = plt.subplots(figsize=(15, 6))
            
            # 基准值设置为0.5（逻辑回归的默认阈值）
            base_value = 0.5
            # 最终预测值
            prediction = model.predict_proba(df)[0][1]
            
            # 设置x轴范围
            # 确定合适的x轴范围
            max_impact = max(abs(np.max(impacts)), abs(np.min(impacts))) * 1.2
            xlim_min = -max_impact
            xlim_max = max_impact
            
            # 设置基准值在图表上的位置
            base_value_pos = 0
            
            # 配置图表参数
            ax.set_xlim(xlim_min, xlim_max)
            ax.set_ylim(-1.2, 0.5)
            
            # 绘制水平基线
            ax.axhline(y=0, color='#888888', linestyle='-', linewidth=1)
            
            # 按影响力排序特征
            feature_impacts = list(zip(feature_names, impacts, feature_values))
            
            # 将特征分为负向影响（蓝色）和正向影响（红色）
            neg_features = [(f, i, v) for f, i, v in feature_impacts if i < 0]
            neg_features = sorted(neg_features, key=lambda x: x[1])  # 从最负向到最不负向排序
            
            pos_features = [(f, i, v) for f, i, v in feature_impacts if i > 0]
            pos_features = sorted(pos_features, key=lambda x: x[1], reverse=True)  # 从最正向到最不正向排序
            
            # 限制显示的特征数量
            max_features = min(8, max(len(neg_features), len(pos_features)))
            neg_features = neg_features[:max_features]
            pos_features = pos_features[:max_features]
            
            # 添加高/低指示器
            ax.text(xlim_max*0.75, 0.35, "higher", ha='center', va='center', color='#ff0051', fontsize=12)
            ax.text(xlim_min*0.75, 0.35, "lower", ha='center', va='center', color='#008bfb', fontsize=12)
            
            # 添加预测值文本
            ax.text(0, 0.35, f"base value\n{base_value:.2f}", ha='center', va='center', color='#888888', fontsize=12)
            ax.text(xlim_max*0.75, 0.15, f"f(x)\n{prediction:.2f}", ha='center', va='center', fontsize=12)
            
            # 绘制连续流图
            current_x = base_value_pos
            x_positions = {}  # 跟踪特征标签位置
            
            # 绘制负向影响（蓝色，向左）
            for i, (feat, impact, val) in enumerate(neg_features):
                # 获取分类特征的原始值
                if feat in ["Sex", "Diabetes_mellitus", "UrineLeuk_bin", "Channel_size", "MayoScore_bin", "degree_of_hydronephrosis"]:
                    orig_val = display_data[feat]
                    display_name = f"{feat}_{orig_val}"
                else:
                    display_name = feat
                
                # 计算段点
                start_x = current_x
                end_x = current_x + impact  # impact为负值
                
                # 绘制梯形
                height = 0.15
                verts = [
                    (start_x, -height),  # 左下
                    (end_x, -height),    # 右下
                    (end_x, height),     # 右上
                    (start_x, height)    # 左上
                ]
                
                # 创建梯形的多边形
                trap = plt.Polygon(verts, closed=True, fill=True, 
                                  facecolor='#008bfb', alpha=0.6, edgecolor=None)
                ax.add_patch(trap)
                
                # 创建箭头形状
                arrow_width = min(abs(impact) * 0.3, 0.5)
                arrow_verts = [
                    (end_x, -height),
                    (end_x - arrow_width, 0),
                    (end_x, height)
                ]
                arrow = plt.Polygon(arrow_verts, closed=True, fill=True,
                                   facecolor='#008bfb', alpha=0.8, edgecolor=None)
                ax.add_patch(arrow)
                
                # 记录标签位置
                x_positions[feat] = (start_x + end_x) / 2
                
                # 更新当前位置
                current_x = end_x
            
            # 重置为基准值，处理正向特征
            current_x = base_value_pos
            
            # 绘制正向影响（红色，向右）
            for i, (feat, impact, val) in enumerate(pos_features):
                # 获取分类特征的原始值
                if feat in ["Sex", "Diabetes_mellitus", "UrineLeuk_bin", "Channel_size", "MayoScore_bin", "degree_of_hydronephrosis"]:
                    orig_val = display_data[feat]
                    display_name = f"{feat}_{orig_val}"
                else:
                    display_name = feat
                
                # 计算段点
                start_x = current_x
                end_x = current_x + impact  # impact为正值
                
                # 绘制梯形
                height = 0.15
                verts = [
                    (start_x, -height),  # 左下
                    (end_x, -height),    # 右下
                    (end_x, height),     # 右上
                    (start_x, height)    # 左上
                ]
                
                # 创建梯形的多边形
                trap = plt.Polygon(verts, closed=True, fill=True,
                                  facecolor='#ff0051', alpha=0.6, edgecolor=None)
                ax.add_patch(trap)
                
                # 创建箭头形状
                arrow_width = min(abs(impact) * 0.3, 0.5)
                arrow_verts = [
                    (end_x, -height),
                    (end_x + arrow_width, 0),
                    (end_x, height)
                ]
                arrow = plt.Polygon(arrow_verts, closed=True, fill=True,
                                   facecolor='#ff0051', alpha=0.8, edgecolor=None)
                ax.add_patch(arrow)
                
                # 记录标签位置
                x_positions[feat] = (start_x + end_x) / 2
                
                # 更新当前位置
                current_x = end_x
            
            # 标记基准值
            ax.plot([base_value_pos], [0], 'o', markersize=8, color='#888888')
            
            # 按位置将特征分组以避免重叠
            pos_threshold = 0.8
            grouped_positions = {}
            
            # 按位置亲近度分组特征
            for feat, pos in sorted(x_positions.items(), key=lambda x: x[1]):
                # 检查这个特征是否靠近已有分组
                found_group = False
                for group_pos, feats in grouped_positions.items():
                    if abs(pos - group_pos) < pos_threshold:
                        feats.append(feat)
                        found_group = True
                        break
                
                # 如果不靠近任何分组，创建新分组
                if not found_group:
                    grouped_positions[pos] = [feat]
            
            # 显示分组中的标签
            for group_pos, feats in grouped_positions.items():
                if len(feats) == 1:
                    # 单个特征，正常显示
                    feat = feats[0]
                    impact = next(i for f, i, v in feature_impacts if f == feat)
                    color = '#ff0051' if impact > 0 else '#008bfb'
                    
                    # 获取标签文本
                    if feat in ["Sex", "Diabetes_mellitus", "UrineLeuk_bin", "Channel_size", "MayoScore_bin", "degree_of_hydronephrosis"]:
                        orig_val = display_data[feat]
                        display_feat = f"{feat}_{orig_val}"
                    else:
                        display_feat = feat
                    
                    # 添加特征名称和影响值
                    ax.text(group_pos, -0.35, display_feat, ha='center', va='center', fontsize=10, color=color)
                    ax.text(group_pos, -0.5, f"{impact:.2f}", ha='center', va='center', fontsize=10, color=color)
                
                else:
                    # 多个特征，需要水平和垂直偏移
                    for i, feat in enumerate(feats):
                        impact = next(i for f, i, v in feature_impacts if f == feat)
                        color = '#ff0051' if impact > 0 else '#008bfb'
                        
                        # 获取标签文本
                        if feat in ["Sex", "Diabetes_mellitus", "UrineLeuk_bin", "Channel_size", "MayoScore_bin", "degree_of_hydronephrosis"]:
                            orig_val = display_data[feat]
                            display_feat = f"{feat}_{orig_val}"
                        else:
                            display_feat = feat
                        
                        # 计算水平偏移
                        offset = (i - (len(feats) - 1) / 2) * 1.2
                        
                        # 对多特征分组使用交错式垂直布局
                        if len(feats) > 3:
                            vert_offset = -0.35 + (i % 2) * -0.25  # 垂直交错
                            value_offset = vert_offset - 0.15  # 数值显示在名称下方
                        else:
                            vert_offset = -0.35
                            value_offset = -0.5
                        
                        # 添加特征名称和影响值
                        ax.text(group_pos + offset, vert_offset, display_feat, ha='center', va='center', fontsize=10, color=color)
                        ax.text(group_pos + offset, value_offset, f"{impact:.2f}", ha='center', va='center', fontsize=10, color=color)
            
            # 移除y轴刻度和边框
            ax.set_yticks([])
            for spine in ['top', 'right', 'left']:
                ax.spines[spine].set_visible(False)
            
            # 添加标题
            ax.set_title(f"Features pushing prediction from base value ({base_value:.2f}) to {prediction:.2f}", fontsize=14)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # 解释
            st.subheader("How to interpret this visualization")
            st.markdown("""
            - **Red bars** push the prediction **higher** (increase fever risk)
            - **Blue bars** push the prediction **lower** (decrease fever risk)
            - The prediction starts at the base value (0.5) and each feature contributes to moving it toward the final prediction
            - The wider the bar, the stronger the impact of that feature
            """)
            
            # 显示前5个最有影响力的特征
            st.subheader("Top Feature Impacts")
            st.markdown("Key factors affecting this prediction:")
            
            # 按影响力绝对值排序
            sorted_features = sorted(feature_impacts, key=lambda x: abs(x[1]), reverse=True)
            
            for feat, impact, val in sorted_features[:5]:
                direction = "increases" if impact > 0 else "decreases"
                if feat in ["Sex", "Diabetes_mellitus", "UrineLeuk_bin", "Channel_size", "MayoScore_bin", "degree_of_hydronephrosis"]:
                    orig_val = display_data[feat]
                    st.markdown(f"- **{feat} = {orig_val}**: {direction} fever risk (impact: {impact:.4f})")
                else:
                    st.markdown(f"- **{feat} = {val:.2f}**: {direction} fever risk (impact: {impact:.4f})")
                
        except Exception as e:
            st.warning(f"Could not generate feature impact visualization: {str(e)}")
            
            # 简单的备用可视化
            try:
                st.subheader("Feature Importance (Basic Visualization)")
                
                # 使用模型系数作为特征重要性
                coeffs = model.coef_[0]
                feature_names = df.columns.tolist()
                
                # 创建简单条形图
                plt.figure(figsize=(10, 6))
                colors = ['#ff0051' if c > 0 else '#008bfb' for c in coeffs]
                plt.barh(feature_names, np.abs(coeffs), color=colors)
                plt.xlabel('Absolute Coefficient Value')
                plt.title('Feature Importance for Fever Risk')
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