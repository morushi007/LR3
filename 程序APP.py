# -*- coding: utf-8 -*-
"""
Created on Tue May 13 21:08:04 2025

@author: LENOVO
"""

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
    
    # Calculate the range of impacts to ensure the plot spans appropriately
    total_pos_impact = sum([i for i in impacts if i > 0])
    total_neg_impact = sum([i for i in impacts if i < 0])
    
    # Setup x-axis with more dynamic range based on actual impacts
    # Add 20% padding to ensure all elements are visible
    padding = 0.2
    xlim_min = min(-4 + total_neg_impact * (1 + padding), -10)
    xlim_max = max(-4 + total_pos_impact * (1 + padding), 4)
    base_value_pos = -4  # Position of base value on x-axis
    
    # Set up plot parameters
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(-1.5, 0.7)  # Increased bottom margin for labels to accommodate staggered text
    
    # Draw horizontal line for x-axis
    ax.axhline(y=0, color='#888888', linestyle='-', linewidth=1)
    
    # Add x-axis ticks - dynamically generate based on the range
    x_ticks = list(range(int(xlim_min), int(xlim_max) + 1, 2))
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
    
    # Calculate positions for higher/lower indicators based on the plot range
    higher_pos = xlim_max - 2
    lower_pos = xlim_max - 1
    
    # Add higher/lower indicators
    ax.text(higher_pos, 0.45, "higher", ha='center', va='center', color='#ff0051', fontsize=10)
    ax.text(lower_pos, 0.45, "lower", ha='center', va='center', color='#008bfb', fontsize=10)
    
    # Add f(x) value - position it near the higher/lower indicators
    fx_pos = (higher_pos + lower_pos) / 2
    ax.text(fx_pos, 0.6, f"f(x)\n{prediction:.2f}", ha='center', va='center', fontsize=10)
    
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
    
    # Rest of the code continues as before...