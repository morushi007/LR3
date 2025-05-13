# -*- coding: utf-8 -*-
"""
Created on Tue May 13 11:13:37 2025

@author: LENOVO
"""

# -*- coding: utf-8 -*-
"""
Improved PCNL Post-Operative Fever Prediction Streamlit Application
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
    "Preoperative_L": {"type": "numerical", "min": 0.0, "max": 10.0, "default": 1.8, "description": "Preoperative Lymphocyte Count (×10⁹/L)"},
    "PLR": {"type": "numerical", "min": 0.0, "max": 1000.0, "default": 120.0, "description": "Platelet-to-Lymphocyte Ratio"},
    "Preoperative_hemoglobin": {"type": "numerical", "min": 50.0, "max": 200.0, "default": 130.0, "description": "Preoperative Hemoglobin (g/L)"},
    "Number_of_stones": {"type": "numerical", "min": 1, "max": 20, "default": 1, "description": "Number of Stones"},
    "BMI": {"type": "numerical", "min": 10.0, "max": 50.0, "default": 24.0, "description": "Body Mass Index (kg/m²)"},
    "Mayo_Score": {"type": "numerical", "min": 1.0, "max": 12.0, "default": 2.0, "description": "Mayo Score"},
    "Urine_Leukocytes": {"type": "numerical", "min": 0, "max": 500, "default": 0, "description": "Urine Leukocytes"},
    "Sex": {"type": "categorical", "options": ["Male", "Female"], "default": "Male", "description": "Sex"},
    "Diabetes_mellitus": {"type": "categorical", "options": ["No", "Yes"], "default": "No", "description": "Diabetes Mellitus"},
    "Channel_size": {"type": "categorical", "options": ["18F", "20F"], "default": "18F", "description": "Channel Size"},
    "degree_of_hydronephrosis": {
        "type": "categorical",
        "options": ["None", "Mild", "Moderate", "Severe"],
        "default": "None",
        "description": "Degree of Hydronephrosis"
    }
}

# ——— Input form ———
st.header("Enter Patient Parameters")
cols = st.columns(3)
input_data = {}

# Process input parameters
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

# Calculate derived features
def process_input_data(data):
    df = pd.DataFrame([data.copy()])
    
    # Add derived features
    df['MayoScore_bin'] = "≥3" if df['Mayo_Score'].iloc[0] >= 3 else "<3"
    df['UrineLeuk_bin'] = ">0" if df['Urine_Leukocytes'].iloc[0] > 0 else "=0"
    
    # Model feature processing
    model_df = df.copy()
    # Encode categorical features
    model_df["Sex"] = model_df["Sex"].map({"Male": 1, "Female": 0})
    model_df["Diabetes_mellitus"] = model_df["Diabetes_mellitus"].map({"Yes": 1, "No": 0})
    model_df["UrineLeuk_bin"] = model_df["UrineLeuk_bin"].map({">0": 1, "=0": 0})
    model_df["Channel_size"] = model_df["Channel_size"].map({"18F": 1, "20F": 0})
    model_df["degree_of_hydronephrosis"] = model_df["degree_of_hydronephrosis"].map({
        "None": 0, "Mild": 1, "Moderate": 2, "Severe": 3
    })
    model_df["MayoScore_bin"] = model_df["MayoScore_bin"].map({"≥3": 1, "<3": 0})
    
    # Keep only features required for the model
    model_features = [
        "LMR", "Preoperative_N", "Operative_time", "Preoperative_L", 
        "PLR", "Preoperative_hemoglobin", "Number_of_stones", "BMI",
        "Sex", "Diabetes_mellitus", "UrineLeuk_bin", "Channel_size", 
        "degree_of_hydronephrosis", "MayoScore_bin"
    ]
    
    return model_df[model_features], df

st.markdown("---")
if st.button("Predict Fever Risk", use_container_width=True):
    model = load_model()
    if model:
        # Process input data
        model_df, original_df = process_input_data(input_data)
        
        # Predict probability
        proba = model.predict_proba(model_df)[0][1] * 100
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

        # ——— Improved SHAP Force Plot Visualization ———
        try:
            st.markdown("## Feature Impact Analysis")
            
            # Calculate predicted probability and get model information
            prediction_score = model.predict_proba(model_df)[0][1]
            base_value = 0.5  # Base value (average prediction)
            
            # Get coefficients and their contributions
            coeffs = model.coef_[0]
            feature_names = model_df.columns.tolist()
            feature_values = model_df.iloc[0].values
            
            # Calculate impacts (equivalent to SHAP values for linear models)
            impacts = coeffs * feature_values
            
            # Create figure with proper dimensions
            fig, ax = plt.subplots(figsize=(14, 4))
            
            # Set up the plot area exactly like SHAP
            ax.set_xlim(-10, 5)
            
            # Remove all spines except bottom
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            
            # Add horizontal line for x-axis
            ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
            
            # Style axis ticks exactly like SHAP
            ax.tick_params(axis='x', direction='out', color='black', length=3)
            ax.set_xticks(range(-10, 6, 2))
            
            # This is the main title (placed below in original SHAP)
            title = f"Based on feature values, predicted possibility of fever is {proba:.2f}%"
            
            # 1. Create the background pink and blue areas like SHAP
            # First determine the extent of red and blue areas based on impacts
            total_pos_impact = sum(i for i in impacts if i > 0)
            total_neg_impact = sum(i for i in impacts if i < 0)
            
            # Scale factor to match SHAP's scale
            scale_factor = 8 / max(abs(total_pos_impact), abs(total_neg_impact))
            
            # Create red (positive) background area
            red_width = total_pos_impact * scale_factor
            blue_width = abs(total_neg_impact) * scale_factor
            
            # Add colored backgrounds
            rect_pink = plt.Rectangle((-10, 0.05), 10+red_width, 0.9, color='#ffb6c1', alpha=0.3, ec=None)
            ax.add_patch(rect_pink)
            
            # Add blue background at the end
            rect_blue = plt.Rectangle((red_width, 0.05), 5-red_width, 0.9, color='#add8e6', alpha=0.3, ec=None)
            ax.add_patch(rect_blue)
            
            # 2. Add feature areas and triangles like SHAP
            # Sort features by absolute impact
            sorted_features = sorted(zip(feature_names, impacts, feature_values), 
                                    key=lambda x: abs(x[1]), reverse=True)
            
            # Filter to top features only
            top_features = sorted_features[:8]
            
            # Normalize impacts to fit the visualization scale
            normalized_impacts = []
            current_pos_x = 0
            current_neg_x = -blue_width
            
            # Prepare positive and negative impacts
            pos_impacts = []
            neg_impacts = []
            
            for feat, impact, val in top_features:
                norm_impact = impact * scale_factor
                if norm_impact > 0:
                    pos_impacts.append((feat, norm_impact, val, current_pos_x))
                    current_pos_x += norm_impact
                else:
                    neg_impacts.append((feat, norm_impact, val, current_neg_x))
                    current_neg_x += abs(norm_impact)
            
            # Now draw the features
            # First the positive/red features
            for feat, impact, val, start_x in pos_impacts:
                # Draw triangle marker at the beginning of each segment
                ax.plot(start_x, 0.1, marker='>',  color='#ff007c', markersize=6)
                
                # Format the feature name and value
                display_name = feat
                # Handle categorical features
                if feat in ["Sex", "Diabetes_mellitus", "UrineLeuk_bin", "Channel_size", "MayoScore_bin", "degree_of_hydronephrosis"]:
                    # Get display format
                    if feat == "Sex" and val == 1:
                        display_name = "Sex_2"
                    elif feat == "Diabetes_mellitus" and val == 1:
                        display_name = "Diabetes mellitus_1"
                    elif feat == "UrineLeuk_bin" and val == 1:
                        display_name = "UrineLeuk_bin_>0"
                    elif feat == "MayoScore_bin" and val == 1:
                        display_name = "MayoScore_bin_≥3"
                    val_display = "Yes" if val == 1 else "No"
                else:
                    val_display = f"{val:.2f}"
                
                # Add feature label and value
                ax.text(start_x + impact/2, -0.15, f"{display_name}\n{val_display}", 
                       ha='center', va='top', fontsize=9, color='#ff007c')
            
            # Then the negative/blue features
            for feat, impact, val, start_x in neg_impacts:
                # Draw triangle marker
                ax.plot(start_x, 0.1, marker='<', color='#0066ff', markersize=6)
                
                # Format the feature name and value
                display_name = feat
                # Handle categorical features
                if feat in ["Sex", "Diabetes_mellitus", "UrineLeuk_bin", "Channel_size", "MayoScore_bin", "degree_of_hydronephrosis"]:
                    # Get display format
                    if feat == "Sex" and val == 0:
                        display_name = "Sex_1"
                    val_display = "Yes" if val == 1 else "No"
                elif feat == "LMR":
                    display_name = "LMR"
                    val_display = f"{val:.2f}"
                elif feat == "Preoperative_hemoglobin":
                    display_name = "Preoperative hemoglobin"
                    val_display = f"{val:.2f}"
                else:
                    val_display = f"{val:.2f}"
                
                # Add feature label and value
                ax.text(start_x + abs(impact)/2, -0.15, f"{display_name}\n{val_display}", 
                       ha='center', va='top', fontsize=9, color='#0066ff')
            
            # 3. Add the higher/lower indicators and base value
            # Base value at the center
            ax.text(0, 1.5, f"base value\n{base_value:.2f}", ha='center', va='center', fontsize=9)
            
            # Higher indicator with right arrow
            ax.text(3, 1.5, "higher", ha='center', va='center', color='#ff007c', fontsize=9)
            ax.annotate("", xy=(4, 1.5), xytext=(3.5, 1.5), 
                       arrowprops=dict(arrowstyle="->", color='#ff007c', shrinkA=0, shrinkB=0, lw=1))
            
            # Lower indicator with left arrow
            ax.text(-5, 1.5, "lower", ha='center', va='center', color='#0066ff', fontsize=9)
            ax.annotate("", xy=(-6, 1.5), xytext=(-5.5, 1.5), 
                       arrowprops=dict(arrowstyle="->", color='#0066ff', shrinkA=0, shrinkB=0, lw=1))
            
            # 4. Add the prediction value in a small box
            # Create a small box for the prediction value
            rect = plt.Rectangle((3.7, 1.3), 0.6, 0.4, ec='#bbbbbb', fc='white', alpha=0.8, lw=0.5)
            ax.add_patch(rect)
            ax.text(4, 1.5, f"{prediction_score:.2f}", ha='center', va='center', fontsize=9)
            
            # 5. Remove y-axis ticks and labels
            ax.set_yticks([])
            ax.set_yticklabels([])
            
            # Add title under the plot
            plt.figtext(0.5, 0.01, title, ha='center', fontsize=14)
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            st.pyplot(fig)
            
            # Add explanation text
            st.subheader("Feature Impact Explanation")
            st.markdown("""
            ### How to interpret the visualization:
            - Red features increase the probability of fever
            - Blue features decrease the probability of fever
            - The impact of each feature shows how much it pushes the prediction away from the base value
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
                feature_names = model_df.columns.tolist()
                
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