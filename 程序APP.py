# -*- coding: utf-8 -*-
"""
Created on Tue May 13 10:29:23 2025

@author: LENOVO
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 13 10:18:25 2025
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
from matplotlib.patches import Rectangle

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
    "degree_of_hydronephrosis": {"type": "categorical", "options": ["None", "Mild", "Moderate", "Severe"], "default": "None", "description": "Degree of Hydronephrosis"},
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
                min_value=cfg["min"], max_value=cfg["max"],
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
        # encode categoricals
        df["Sex"] = df["Sex"].map({"Male": 1, "Female": 0})
        df["Diabetes_mellitus"] = df["Diabetes_mellitus"].map({"Yes": 1, "No": 0})
        df["UrineLeuk_bin"] = df["UrineLeuk_bin"].map({">0": 1, "=0": 0})
        df["Channel_size"] = df["Channel_size"].map({"18F": 1, "20F": 0})
        df["degree_of_hydronephrosis"] = df["degree_of_hydronephrosis"].map({"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3})
        df["MayoScore_bin"] = df["MayoScore_bin"].map({"≥3": 1, "<3": 0})

        # predict
        proba = model.predict_proba(df)[0][1] * 100
        if proba < 25:
            level, color = "Low Risk", "green"
        elif proba < 50:
            level, color = "Moderate-Low Risk", "lightgreen"
        elif proba < 75:
            level, color = "Moderate-High Risk", "orange"
        else:
            level, color = "High Risk", "red"

        # display main results
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

        # ——— Improved SHAP-style Force Plot ———
        try:
            st.markdown("## Feature Impact Analysis")
            # prepare data
            coeffs = model.coef_[0]
            feature_names = df.columns.tolist()
            feature_values = df.iloc[0].values
            impacts = coeffs * feature_values
            feat_imp = list(zip(feature_names, impacts, feature_values))
            # sort by absolute impact and take top 6
            top_feats = sorted(feat_imp, key=lambda x: abs(x[1]), reverse=True)[:6]
            # normalization factor
            max_imp = max(abs(i) for _, i, _ in feat_imp)
            norm = 9 / max_imp if max_imp > 0 else 1

            # draw
            fig, ax = plt.subplots(figsize=(10, 2.5))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
            ax.axvline(0, color='#999999', linestyle='-')
            ax.set_title(f"Based on feature values, predicted fever risk is {proba:.2f}%", fontsize=14)

            neg_cum, pos_cum = 0, 0
            categoricals = {"Sex","Diabetes_mellitus","UrineLeuk_bin","Channel_size","degree_of_hydronephrosis","MayoScore_bin"}

            for feat, raw_imp, val in top_feats:
                imp = raw_imp * norm
                color = '#ff0051' if imp > 0 else '#008bfb'
                if imp > 0:
                    start, width = pos_cum, imp
                    pos_cum += width
                else:
                    width = abs(imp)
                    start = -neg_cum - width
                    neg_cum += width

                ax.add_patch(Rectangle((start, 0.1), width, 0.8, color=color, alpha=0.7))

                # label
                lbl = input_data[feat] if feat in categoricals else f"{val:.2f}"
                ax.text(start + width/2, -0.2, f"{feat}\n{lbl}", ha='center', va='top', fontsize=9)

            # adjust limits and add higher/lower
            span = max(pos_cum, neg_cum) * 1.1
            ax.set_xlim(-span, span)
            ax.text(-span, 1.3, "lower", ha='left', color='#008bfb', fontsize=10)
            ax.text(span, 1.3, "higher", ha='right', color='#ff0051', fontsize=10)
            # show final prediction offset
            base = 0.5
            final_pos = (model.predict_proba(df)[0][1] - base) * norm
            ax.text(final_pos, 1.3, f"{model.predict_proba(df)[0][1]:.2f}", ha='center', fontsize=10)

            plt.tight_layout()
            st.pyplot(fig)

            st.subheader("Feature Contribution Table")
            contrib_df = pd.DataFrame({
                'Feature': feature_names,
                'Value': feature_values,
                'Impact': impacts,
                'Direction': ['↑ fever risk' if i>0 else '↓ fever risk' for i in impacts]
            }).sort_values('Impact', key=abs, ascending=False)
            st.dataframe(contrib_df)

        except Exception as e:
            st.warning(f"Could not generate enhanced force plot: {e}")

            # fallback simple importance bar chart
            st.subheader("Feature Importance (Fallback)")
            fig, ax = plt.subplots(figsize=(10, 6))
            cols = ['#ff0051' if c>0 else '#008bfb' for c in coeffs]
            ax.barh(feature_names, np.abs(coeffs), color=cols)
            ax.set_xlabel('Absolute Coefficient Value')
            ax.set_title('Feature Impact on Fever Risk')
            plt.tight_layout()
            st.pyplot(fig)

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
