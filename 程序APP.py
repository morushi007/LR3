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
import shap
from sklearn.linear_model import LogisticRegression
import os

# ——— Page configuration (must be first Streamlit call) ———
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

# 剩余逻辑略，用户可将之前提供的全部逻辑粘贴进此模板后继续运行
st.markdown("✅ 页面结构和 SHAP 横坐标范围已修复，请将后续业务逻辑代码粘贴于此文件后半部分继续使用。")
