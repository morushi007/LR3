# -*- coding: utf-8 -*-
"""
Created on Tue May 13 12:49:01 2025

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
            st.markdown("""
            <div style="padding:10px;border-radius:5px;background-color:#f0f2f6;">
                <p style="margin-bottom:0;"><strong>Red bars increase fever risk; blue bars decrease risk.</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Get model coefficients and feature values
            coeffs = model.coef_[0]
            feature_names = df.columns.tolist()
            feature_values = df.iloc[0].values
            
            # Calculate feature impacts (coefficient * value)
            impacts = coeffs * feature_values
            
            # Create the SHAP-style force plot (waterfall style)
            fig, ax = plt.subplots(figsize=(12, 5))
            
            # Base value (typically set at 0.5 for logistic regression probability)
            base_value = 0.5
            prediction = model.predict_proba(df)[0][1]
            
            # Sort features by impact (not by absolute value)
            # This creates a more coherent visualization where we see progression from negative to positive
            feature_impacts = list(zip(feature_names, impacts, feature_values))
            
            # We'll sort features by their impact, but we want negative impacts first
            neg_features = sorted([(f, i, v) for f, i, v in feature_impacts if i < 0], key=lambda x: x[1])
            pos_features = sorted([(f, i, v) for f, i, v in feature_impacts if i > 0], key=lambda x: x[1], reverse=True)
            
            # Create a list of x-axis positions for plotting
            x_ticks = [-10, -8, -6, -4, -2, 0, 2, 4]
            
            # Draw the x-axis
            ax.axhline(y=0, color='#888888', linestyle='-', linewidth=1, zorder=1)
            
            # Set up the general plot parameters
            ax.set_xlim(min(x_ticks), max(x_ticks))
            ax.set_ylim(-0.8, 0.8)
            ax.set_yticks([])
            
            # Add tick marks to x-axis
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([str(x) for x in x_ticks], fontsize=8)
            
            # Create a color gradient for positive and negative features
            # Using semi-transparent colors for the areas
            red_color = '#ff0051'
            blue_color = '#008bfb'
            
            # Mark the base value
            base_x = -4  # Position on x-axis
            ax.text(base_x, 0.1, "base value", ha='center', va='bottom', fontsize=10, color='#888888')
            ax.plot([base_x], [0], 'o', color='#888888', markersize=8)
            
            # Draw the prediction marker and label
            pred_x = 2.9  # Estimated position for the prediction
            ax.text(pred_x, 0.35, "f(x)\n" + f"{prediction:.2f}", ha='center', va='bottom', fontsize=10)
            
            # Add direction labels
            ax.text(1.7, 0.35, "higher", ha='center', va='bottom', color=red_color, fontsize=10)
            ax.text(3.8, 0.35, "lower", ha='center', va='bottom', color=blue_color, fontsize=10)
            
            # Draw the feature contributions
            # First, we'll draw arrowheads to show direction and magnitude
            y_height = -0.4  # Height for feature labels
            
            # Now we start plotting features as arrows with labels
            # Define arrow properties
            arrow_props = dict(
                arrowstyle="wedge,tail_width=0.7",
                shrinkA=0,
                shrinkB=0,
                linewidth=0,
                zorder=3
            )
            
            # Draw arrow for base value
            current_x = base_x
            
            # Draw positive features (pushing prediction higher)
            for i, (feat, impact, val) in enumerate(pos_features):
                # Scale impact for visualization
                scaled_impact = impact * 0.8  # Adjust scaling factor as needed
                
                # Determine start and end points for arrow
                start_x = current_x
                end_x = current_x + scaled_impact
                
                # Draw arrow (positive impact)
                ax.add_patch(plt.Polygon(
                    [[start_x, 0], [end_x, 0], [end_x, 0.4]],
                    closed=True, 
                    facecolor=red_color, 
                    alpha=0.7
                ))
                
                # Add feature label and value
                if feat in ["Sex", "Diabetes_mellitus", "UrineLeuk_bin", "Channel_size", "MayoScore_bin", "degree_of_hydronephrosis"]:
                    # For categorical features, show the original value
                    orig_val = input_data[feat]
                    label_text = f"{feat}_{orig_val}"
                else:
                    label_text = f"{feat}"
                    
                # Show value as a separate line
                value_text = f"{scaled_impact:.2f}"
                
                # Position labels below the arrow at consistent height
                ax.text(start_x + scaled_impact/2, y_height, label_text, 
                        ha='center', va='bottom', fontsize=8, color=red_color, rotation=0)
                ax.text(start_x + scaled_impact/2, y_height-0.1, value_text, 
                        ha='center', va='bottom', fontsize=8, color=red_color)
                
                # Update current position
                current_x = end_x
            
            # Reset to base value for negative features
            current_x = base_x
            
            # Draw negative features (pushing prediction lower)
            for i, (feat, impact, val) in enumerate(neg_features):
                # Scale impact for visualization (absolute value as it's negative)
                scaled_impact = impact * 0.8  # Adjust scaling factor as needed
                
                # Determine start and end points for arrow
                start_x = current_x
                end_x = current_x + scaled_impact  # Note: impact is already negative
                
                # Draw arrow (negative impact)
                ax.add_patch(plt.Polygon(
                    [[start_x, 0], [end_x, 0], [end_x, 0.4]],
                    closed=True, 
                    facecolor=blue_color, 
                    alpha=0.7
                ))
                
                # Add feature label and value
                if feat in ["Sex", "Diabetes_mellitus", "UrineLeuk_bin", "Channel_size", "MayoScore_bin", "degree_of_hydronephrosis"]:
                    # For categorical features, show the original value
                    orig_val = input_data[feat]
                    label_text = f"{feat}_{orig_val}"
                else:
                    label_text = f"{feat}"
                    
                # Show value as a separate line
                value_text = f"{scaled_impact:.2f}"
                
                # Position labels below the arrow at consistent height
                ax.text(start_x + scaled_impact/2, y_height, label_text, 
                        ha='center', va='bottom', fontsize=8, color=blue_color, rotation=0)
                ax.text(start_x + scaled_impact/2, y_height-0.1, value_text, 
                        ha='center', va='bottom', fontsize=8, color=blue_color)
                
                # Update current position
                current_x = end_x
            
            # Remove y-axis ticks and spines
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            # Add title
            ax.set_title(f"Features pushing prediction from base value (0.50) to {prediction:.2f}", fontsize=12)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Explanation
            st.subheader("How to interpret this visualization")
            st.markdown("""
            - **Red bars** push the prediction **higher** (increase fever risk)
            - **Blue bars** push the prediction **lower** (decrease fever risk)
            - The prediction starts at the base value (0.5) and each feature contributes to moving it toward the final prediction
            - The wider the bar, the stronger the impact of that feature
            """)
            
            # Display feature contributions table
            st.subheader("All Feature Contributions")
            feature_contrib = pd.DataFrame({
                'Feature': feature_names,
                'Value': [input_data[f] if f in ["Sex", "Diabetes_mellitus", "UrineLeuk_bin", "Channel_size", "MayoScore_bin", "degree_of_hydronephrosis"] else feature_values[i] for i, f in enumerate(feature_names)],
                'Impact': impacts,
                'Direction': ['Increases fever risk' if i > 0 else 'Decreases fever risk' for i in impacts]
            }).sort_values('Impact', key=abs, ascending=False)
            
            st.dataframe(feature_contrib)
            
            # Display the top 5 influential features as text
            st.subheader("Top Feature Impacts")
            st.markdown("Key factors affecting prediction:")
            for feat, impact, val in sorted_features[:5]:
                direction = "increases" if impact > 0 else "decreases"
                if feat in ["Sex", "Diabetes_mellitus", "UrineLeuk_bin", "Channel_size", "MayoScore_bin", "degree_of_hydronephrosis"]:
                    orig_val = input_data[feat]
                    st.markdown(f"- **{feat} = {orig_val}**: {direction} fever risk")
                else:
                    st.markdown(f"- **{feat} = {val:.2f}**: {direction} fever risk")
                
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