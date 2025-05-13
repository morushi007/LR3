# -*- coding: utf-8 -*-
"""
Created on Tue May 13 10:18:25 2025

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

        # ——— SHAP Force Plot visualization ———
        try:
            st.markdown("## Feature Impact Analysis")
            
            # Calculate predicted probability and get model information
            prediction_score = model.predict_proba(df)[0][1]
            base_value = 0.5  # Base value (average prediction)
            
            # Get coefficients and their contributions
            coeffs = model.coef_[0]
            feature_names = df.columns.tolist()
            feature_values = df.iloc[0].values
            
            # Calculate impacts - for logistic regression, use coefficients * values
            impacts = coeffs * feature_values
            
            # Create SHAP-style force plot
            fig, ax = plt.subplots(figsize=(10, 2.5))
            
            # Set up the plot area
            ax.set_xlim(-10, 10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            # Add vertical center line
            ax.axvline(x=0, color='#999999', linestyle='-')
            
            # Add title
            ax.set_title(f"Based on feature values, predicted possibility of fever is {proba:.2f}%", fontsize=14)
            
            # Standardize and sort feature impacts
            feature_impacts = list(zip(feature_names, impacts, feature_values))
            sorted_features = sorted(feature_impacts, key=lambda x: abs(x[1]), reverse=True)
            
            # Select top features for display
            top_features = sorted_features[:6]  # Limit to prevent overcrowding
            
            # Calculate scale factor for normalization
            max_abs_impact = max([abs(imp) for _, imp, _ in sorted_features])
            norm_factor = 9 / max_abs_impact if max_abs_impact > 0 else 1
            
            # Separate positive and negative impacts
            pos_impacts = [(f, i*norm_factor, v) for f, i, v in top_features if i > 0]
            neg_impacts = [(f, i*norm_factor, v) for f, i, v in top_features if i < 0]
            
            # Create force plot visualization with SHAP colors
            # Create main red area (positive impacts)
            pos_width = sum(abs(i) for _, i, _ in pos_impacts) 
            if pos_width > 0:
                rect_red = plt.Rectangle((-pos_width, 0.1), pos_width, 0.8, color='#ff0051', alpha=0.7)
                ax.add_patch(rect_red)
            
            # Create blue area (negative impacts)
            neg_width = sum(abs(i) for _, i, _ in neg_impacts)
            if neg_width > 0:
                rect_blue = plt.Rectangle((0, 0.1), neg_width, 0.8, color='#008bfb', alpha=0.7)
                ax.add_patch(rect_blue)
            
            # Add feature labels
            # Calculate positions for feature labels
            all_display_features = pos_impacts + neg_impacts
            # Sort features by absolute impact for consistent display
            all_display_features = sorted(all_display_features, key=lambda x: abs(x[1]), reverse=True)
            
            # Calculate label positions to spread evenly
            label_positions = np.linspace(-8, 8, len(all_display_features) + 2)[1:-1]
            
            # Display feature labels
            for i, (feat, impact, val) in enumerate(all_display_features):
                # Format label based on feature type
                if feat in ["Sex", "Diabetes_mellitus", "UrineLeuk_bin", "Channel_size", "MayoScore_bin", "degree_of_hydronephrosis"]:
                    orig_val = input_data[feat]
                    ax.text(label_positions[i], -0.2, f"{feat}\n{orig_val}", ha='center', va='top', fontsize=9)
                else:
                    ax.text(label_positions[i], -0.2, f"{feat}\n{val:.2f}", ha='center', va='top', fontsize=9)
            
            # Add base value and prediction value indicators
            ax.text(0, 1.3, f"base value\n{base_value:.2f}", ha='center', fontsize=10)
            
            # Add higher/lower indicators
            ax.text(8, 1.3, "higher", ha='center', color='#ff0051', fontsize=10)
            ax.text(-8, 1.3, "lower", ha='center', color='#008bfb', fontsize=10)
            
            # Add prediction score on the right
            norm_prediction = (prediction_score - base_value) * 10
            ax.text(norm_prediction, 1.3, f"{prediction_score:.2f}", ha='center', fontsize=10)
            
            # Remove y-axis ticks and labels
            ax.set_yticks([])
            ax.set_yticklabels([])
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Add explanation text
            st.subheader("Feature Impact Explanation")
            st.markdown("""
            ### How to interpret the visualization:
            - Red features increase the probability of fever
            - Blue features decrease the probability of fever
            - The longer the bar, the stronger the impact
            """)
            
            # Display feature contributions as a table
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
            
            # Simple fallback visualization 
            try:
                st.subheader("Feature Importance (Basic Visualization)")
                
                # Use model coefficients for feature importance
                coeffs = model.coef_[0]
                feature_names = df.columns.tolist()
                
                # Create simple bar chart
                plt.figure(figsize=(10, 6))
                colors = ['#ff0051' if c > 0 else '#008bfb' for c in coeffs]  # Using SHAP colors
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