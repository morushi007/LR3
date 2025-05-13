# -*- coding: utf-8 -*-
"""
Created on Tue May 13 17:19:27 2025

@author: LENOVO
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 13 16:11:19 2025
Modified to fix streamlit error

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
# IMPORTANT: st.set_page_config() must be the first Streamlit command used in your app
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

        # ——— Feature Impact Analysis ———
        try:
            from matplotlib.path import Path
            from matplotlib.patches import PathPatch

            st.markdown("## Feature Impact Analysis")
            st.markdown("""
            <div style="padding:10px;border-radius:5px;background-color:#f0f2f6;">
                <p style="margin-bottom:0;"><strong>Red bars increase fever risk; blue bars decrease risk.</strong></p>
            </div>
            """, unsafe_allow_html=True)

            coeffs = model.coef_[0]
            feature_names = df.columns.tolist()
            feature_values = df.iloc[0].values
            impacts = coeffs * feature_values
            feature_impacts = list(zip(feature_names, impacts, feature_values))
            sorted_features = sorted(feature_impacts, key=lambda x: abs(x[1]), reverse=True)

            neg_features = [f for f in feature_impacts if f[1] < 0]
            neg_features = sorted(neg_features, key=lambda x: x[1])[:4]
            pos_features = [f for f in feature_impacts if f[1] > 0]
            pos_features = sorted(pos_features, key=lambda x: x[1], reverse=True)[:4]

            base_value_pos = -4

            # dynamic xlim_max
            total_pos = sum(impact for _, impact, _ in pos_features)
            xlim_min = -10
            xlim_max = base_value_pos + total_pos + 0.5

            fig, ax = plt.subplots(figsize=(15, 6))
            ax.set_xlim(xlim_min, xlim_max)
            ax.set_ylim(-1.5, 0.7)
            ax.axhline(y=0, color='#888888', linestyle='-', linewidth=1)
            x_ticks = list(range(xlim_min, int(xlim_max) + 1, 2))
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([str(x) for x in x_ticks], fontsize=8)

            ax.text(2, 0.45, "higher", ha='center', va='center', color='#ff0051', fontsize=10)
            ax.text(3, 0.45, "lower", ha='center', va='center', color='#008bfb', fontsize=10)
            prediction = model.predict_proba(df)[0][1]
            ax.text(2.5, 0.6, f"f(x)\n{prediction:.2f}", ha='center', va='center', fontsize=10)
            ax.text(base_value_pos, 0.45, "base value", ha='center', va='center', color='#888888', fontsize=10)

            current_x = base_value_pos
            # negative bars
            for feat, impact, val in neg_features:
                start_x = current_x
                end_x = current_x + impact
                trap = plt.Polygon(
                    [(start_x, -0.15), (end_x, -0.15), (end_x, 0.15), (start_x, 0.15)],
                    closed=True, facecolor='#008bfb', alpha=0.6, edgecolor=None
                )
                ax.add_patch(trap)
                arrow = plt.Polygon(
                    [(end_x, -0.15), (end_x - min(abs(impact)*0.3,0.5), 0), (end_x, 0.15)],
                    closed=True, facecolor='#008bfb', alpha=0.8, edgecolor=None
                )
                ax.add_patch(arrow)
                current_x = end_x

            current_x = base_value_pos
            # positive bars
            for feat, impact, val in pos_features:
                start_x = current_x
                end_x = current_x + impact
                trap = plt.Polygon(
                    [(start_x, -0.15), (end_x, -0.15), (end_x, 0.15), (start_x, 0.15)],
                    closed=True, facecolor='#ff0051', alpha=0.6, edgecolor=None
                )
                ax.add_patch(trap)
                arrow = plt.Polygon(
                    [(end_x, -0.15), (end_x + min(abs(impact)*0.3,0.5), 0), (end_x, 0.15)],
                    closed=True, facecolor='#ff0051', alpha=0.8, edgecolor=None
                )
                ax.add_patch(arrow)
                current_x = end_x

            ax.plot([base_value_pos], [0], 'o', markersize=8, color='#888888')

            # labels
            x_positions = {f: (base_value_pos + sum(i for _, i, _ in neg_features[:idx+1]))/2
                           for idx, (f, i, _) in enumerate(neg_features)}
            x_positions.update({
                f: (base_value_pos + sum(i for _, i, _ in pos_features[:idx+1]))/2
                for idx, (f, i, _) in enumerate(pos_features)
            })

            for feat, pos in x_positions.items():
                impact = next(i for f, i, _ in feature_impacts if f == feat)
                color = '#ff0051' if impact > 0 else '#008bfb'
                display_feat = f"{feat}_{input_data[feat]}" if feat in [
                    "Sex","Diabetes_mellitus","UrineLeuk_bin","Channel_size","MayoScore_bin","degree_of_hydronephrosis"
                ] else feat
                ax.text(pos, -0.35, display_feat, ha='center', va='center', fontsize=8, color=color)
                ax.text(pos, -0.5, f"{impact:.2f}", ha='center', va='center', fontsize=8, color=color)

            ax.set_yticks([])
            for spine in ['top','right','left']:
                ax.spines[spine].set_visible(False)
            ax.set_title(f"Features pushing prediction from base value (0.50) to {prediction:.2f}", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)

            # Explanation
            st.subheader("How to interpret this visualization")
            st.markdown("""
            - **Red bars** push the prediction **higher** (increase fever risk)  
            - **Blue bars** push the prediction **lower** (decrease fever risk)  
            - The prediction starts at the base value (0.5) and each feature contributes to moving it  
            - The wider the bar, the stronger the impact of that feature
            """)

            # ——— All Feature Contributions Table ———
            st.subheader("All Feature Contributions")
            feature_contrib = pd.DataFrame({
                'Feature': feature_names,
                'Value': [
                    input_data[f] if f in ["Sex","Diabetes_mellitus","UrineLeuk_bin","Channel_size","MayoScore_bin","degree_of_hydronephrosis"]
                    else feature_values[i]
                    for i, f in enumerate(feature_names)
                ],
                'Impact': impacts,
                'Direction': ['Increases fever risk' if i > 0 else 'Decreases fever risk' for i in impacts]
            }).sort_values('Impact', key=lambda x: abs(x), ascending=False)
            st.dataframe(feature_contrib)

            # ——— Top 8 Feature Impacts ———
            st.subheader("Top Feature Impacts")
            st.markdown("Key factors affecting prediction:")
            for feat, impact, val in sorted_features[:8]:
                direction = "increases" if impact > 0 else "decreases"
                if feat in ["Sex","Diabetes_mellitus","UrineLeuk_bin","Channel_size","MayoScore_bin","degree_of_hydronephrosis"]:
                    orig_val = input_data[feat]
                    st.markdown(f"- **{feat} = {orig_val}**: {direction} fever risk")
                else:
                    st.markdown(f"- **{feat} = {val:.2f}**: {direction} fever risk")

        except Exception as e:
            st.warning(f"Could not generate feature impact visualization: {str(e)}")
            # fallback simple bar chart
            try:
                st.subheader("Feature Importance (Basic Visualization)")
                coeffs = model.coef_[0]
                feature_names = df.columns.tolist()
                plt.figure(figsize=(10, 6))
                colors = ['#ff0051' if c > 0 else '#008bfb' for c in coeffs]
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