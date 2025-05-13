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

        # ——— SHAP Force Plot visualization ———
        try:
            # Import needed matplotlib components
            from matplotlib.path import Path
            from matplotlib.patches import PathPatch
            
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
            
            # Create the SHAP-style force plot
            fig, ax = plt.subplots(figsize=(15, 6))  # Increased height from 5 to 6
            
            # Base value and prediction
            base_value = 0.5  # For logistic regression, base value is typically 0.5
            prediction = model.predict_proba(df)[0][1]
            
            # Setup x-axis
            xlim_min = -10
            xlim_max = 4
            base_value_pos = -4  # Position of base value on x-axis
            
            # Set up plot parameters
            ax.set_xlim(xlim_min, xlim_max)
            ax.set_ylim(-1.5, 0.7)  # Increased bottom margin for labels to accommodate staggered text
            
            # Draw horizontal line for x-axis
            ax.axhline(y=0, color='#888888', linestyle='-', linewidth=1)
            
            # Add x-axis ticks
            x_ticks = [-10, -8, -6, -4, -2, 0, 2, 4]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([str(x) for x in x_ticks], fontsize=8)
            
            # Sort features by impact for visualization
            feature_impacts = list(zip(feature_names, impacts, feature_values))
            sorted_features = sorted(feature_impacts, key=lambda x: abs(x[1]), reverse=True)
            
            # First get sorted negative impacts (pushing prediction lower)
            neg_features = [(f, i, v) for f, i, v in feature_impacts if i < 0]
            neg_features = sorted(neg_features, key=lambda x: x[1])  # Sort from most negative to least negative
            
            # Then get sorted positive impacts (pushing prediction higher)
            pos_features = [(f, i, v) for f, i, v in feature_impacts if i > 0]
            pos_features = sorted(pos_features, key=lambda x: x[1], reverse=True)  # Sort from most positive to least positive
            
            # Limit to a larger number of features to ensure MayoScore_bin is included
            max_features = 10  # Increased from 8 to 10 to include more features
            
            # Make sure MayoScore_bin is included if it exists in feature_impacts
            has_mayo_score = any(feat == "MayoScore_bin" for feat, _, _ in feature_impacts)
            
            # Keep only the most influential features
            neg_features = neg_features[:min(max_features//2, len(neg_features))]
            pos_features = pos_features[:min(max_features//2, len(pos_features))]
            
            # If MayoScore_bin exists but is not in either list, add it to the appropriate list
            if has_mayo_score:
                mayo_impact = next((i for f, i, v in feature_impacts if f == "MayoScore_bin"), None)
                mayo_value = next((v for f, i, v in feature_impacts if f == "MayoScore_bin"), None)
                
                if mayo_impact is not None and mayo_value is not None:
                    # Check if MayoScore_bin is already in either list
                    in_neg = any(feat == "MayoScore_bin" for feat, _, _ in neg_features)
                    in_pos = any(feat == "MayoScore_bin" for feat, _, _ in pos_features)
                    
                    if not (in_neg or in_pos):
                        # Add to appropriate list based on impact
                        if mayo_impact < 0:
                            neg_features.append(("MayoScore_bin", mayo_impact, mayo_value))
                        else:
                            pos_features.append(("MayoScore_bin", mayo_impact, mayo_value))
            
            # Add higher/lower indicators
            ax.text(2, 0.45, "higher", ha='center', va='center', color='#ff0051', fontsize=10)
            ax.text(3, 0.45, "lower", ha='center', va='center', color='#008bfb', fontsize=10)
            
            # Add f(x) value
            ax.text(2.5, 0.6, f"f(x)\n{prediction:.2f}", ha='center', va='center', fontsize=10)
            
            # Add base value text
            ax.text(base_value_pos, 0.45, "base value", ha='center', va='center', color='#888888', fontsize=10)
            
            # Draw continuous flow plot
            current_x = base_value_pos
            x_positions = {}  # To track feature positions for label placement
            
            # Generate continuous segments
            # For negative values (blue, to the left)
            for i, (feat, impact, val) in enumerate(neg_features):
                # Get the original binary value for categorical features
                if feat in ["Sex", "Diabetes_mellitus", "UrineLeuk_bin", "Channel_size", "MayoScore_bin", "degree_of_hydronephrosis"]:
                    orig_val = input_data[feat]
                    display_name = f"{feat}_{orig_val}"
                else:
                    display_name = feat
                
                # Calculate segment points
                start_x = current_x
                end_x = current_x + impact  # impact is negative
                
                # Draw gradient-filled trapezoid for this feature
                # First create a trapezoid path
                height = 0.15
                verts = [
                    (start_x, -height),  # bottom left
                    (end_x, -height),    # bottom right
                    (end_x, height),     # top right
                    (start_x, height)    # top left
                ]
                
                # Create polygon for main trapezoid
                trap = plt.Polygon(verts, closed=True, fill=True, 
                                  facecolor='#008bfb', alpha=0.6, edgecolor=None)
                ax.add_patch(trap)
                
                # Create chevron/arrow shape at the end using a triangle
                arrow_width = min(abs(impact) * 0.3, 0.5)
                arrow_verts = [
                    (end_x, -height),
                    (end_x - arrow_width, 0),
                    (end_x, height)
                ]
                arrow = plt.Polygon(arrow_verts, closed=True, fill=True,
                                   facecolor='#008bfb', alpha=0.8, edgecolor=None)
                ax.add_patch(arrow)
                
                # Store position for label
                x_positions[feat] = (start_x + end_x) / 2
                
                # Update current position
                current_x = end_x
            
            # Reset to base value for positive features
            current_x = base_value_pos
            
            # For positive values (red, to the right)
            for i, (feat, impact, val) in enumerate(pos_features):
                # Get the original binary value for categorical features
                if feat in ["Sex", "Diabetes_mellitus", "UrineLeuk_bin", "Channel_size", "MayoScore_bin", "degree_of_hydronephrosis"]:
                    orig_val = input_data[feat]
                    display_name = f"{feat}_{orig_val}"
                else:
                    display_name = feat
                
                # Calculate segment points
                start_x = current_x
                end_x = current_x + impact  # impact is positive
                
                # Draw gradient-filled trapezoid for this feature
                # First create a trapezoid path
                height = 0.15
                verts = [
                    (start_x, -height),  # bottom left
                    (end_x, -height),    # bottom right
                    (end_x, height),     # top right
                    (start_x, height)    # top left
                ]
                
                # Create polygon for main trapezoid
                trap = plt.Polygon(verts, closed=True, fill=True,
                                  facecolor='#ff0051', alpha=0.6, edgecolor=None)
                ax.add_patch(trap)
                
                # Create chevron/arrow shape at the end
                arrow_width = min(abs(impact) * 0.3, 0.5)
                arrow_verts = [
                    (end_x, -height),
                    (end_x + arrow_width, 0),
                    (end_x, height)
                ]
                arrow = plt.Polygon(arrow_verts, closed=True, fill=True,
                                   facecolor='#ff0051', alpha=0.8, edgecolor=None)
                ax.add_patch(arrow)
                
                # Store position for label
                x_positions[feat] = (start_x + end_x) / 2
                
                # Update current position
                current_x = end_x
            
            # Mark base value with circle
            ax.plot([base_value_pos], [0], 'o', markersize=8, color='#888888')
            
            # Add feature labels below with improved spacing to avoid overlapping
            # First, group features by similar positions to avoid overlapping
            pos_threshold = 0.8  # Increased from 0.5 to allow more space between groups
            grouped_positions = {}
            
            # Group features by position proximity
            for feat, pos in sorted(x_positions.items(), key=lambda x: x[1]):
                # Find if this feature is close to an existing group
                found_group = False
                for group_pos, feats in grouped_positions.items():
                    if abs(pos - group_pos) < pos_threshold:
                        feats.append(feat)
                        found_group = True
                        break
                
                # If not close to any group, create a new group
                if not found_group:
                    grouped_positions[pos] = [feat]
            
            # Now display labels with appropriate offsets for each group
            for group_pos, feats in grouped_positions.items():
                # Calculate vertical positions for each feature in group
                if len(feats) == 1:
                    # Single feature, display normally
                    feat = feats[0]
                    impact = next(i for f, i, v in feature_impacts if f == feat)
                    color = '#ff0051' if impact > 0 else '#008bfb'
                    
                    # Get label text
                    if feat in ["Sex", "Diabetes_mellitus", "UrineLeuk_bin", "Channel_size", "MayoScore_bin", "degree_of_hydronephrosis"]:
                        orig_val = input_data[feat]
                        display_feat = f"{feat}_{orig_val}"
                    else:
                        display_feat = feat
                    
                    # Add feature name and impact value
                    ax.text(group_pos, -0.35, display_feat, ha='center', va='center', fontsize=8, color=color)
                    ax.text(group_pos, -0.5, f"{impact:.2f}", ha='center', va='center', fontsize=8, color=color)
                
                else:
                    # Multiple features in one position, need to offset horizontally and vertically
                    for i, feat in enumerate(feats):
                        impact = next(i for f, i, v in feature_impacts if f == feat)
                        color = '#ff0051' if impact > 0 else '#008bfb'
                        
                        # Get label text
                        if feat in ["Sex", "Diabetes_mellitus", "UrineLeuk_bin", "Channel_size", "MayoScore_bin", "degree_of_hydronephrosis"]:
                            orig_val = input_data[feat]
                            display_feat = f"{feat}_{orig_val}"
                        else:
                            display_feat = feat
                        
                        # Calculate horizontal offset to prevent overlapping
                        # Increase spacing between features in the same group
                        offset = (i - (len(feats) - 1) / 2) * 1.2  # Increased from 0.8 to 1.2
                        
                        # For groups with many features, use vertical staggering
                        if len(feats) > 3:
                            # Alternate vertical positions
                            vert_offset = -0.35 + (i % 2) * -0.25  # Stagger vertically
                            value_offset = vert_offset - 0.15  # Value appears below name
                        else:
                            vert_offset = -0.35
                            value_offset = -0.5
                        
                        # Add feature name and impact value with offset
                        ax.text(group_pos + offset, vert_offset, display_feat, ha='center', va='center', fontsize=8, color=color)
                        ax.text(group_pos + offset, value_offset, f"{impact:.2f}", ha='center', va='center', fontsize=8, color=color)
            
            # Remove y-axis ticks and axis borders
            ax.set_yticks([])
            for spine in ['top', 'right', 'left']:
                ax.spines[spine].set_visible(False)
            
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
            
            # Display the top 8 influential features as text
            st.subheader("Top Feature Impacts")
            st.markdown("Key factors affecting prediction:")
            for feat, impact, val in sorted_features[:8]:
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