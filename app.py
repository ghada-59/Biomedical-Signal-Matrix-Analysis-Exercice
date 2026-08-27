import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Biomedical Risk Dashboard", page_icon="🧬", layout="wide")

st.title("🧬 Biomedical Signal Matrix & Clinical Risk Dashboard")
st.markdown("Real-time analysis of biomarker variability and composite risk score calculation for intensive care.")

# 1. Raw Data Matrix
D_raw = np.array([
    [4,  8, 12],
    [6, 10, 15],
    [5,  7,  9],
    [3,  6,  9]
])
patients = ['Patient 1', 'Patient 2', 'Patient 3', 'Patient 4']
features = ['Feature A (HRV)', 'Feature B (Glucose)', 'Feature C (RRv)']

df_matrix = pd.DataFrame(D_raw, index=patients, columns=features)

st.sidebar.header("⚙️ Diagnostic Weight Configuration")
w_a = st.sidebar.slider("Weight Feature A (HRV)", 1, 20, 5)
w_b = st.sidebar.slider("Weight Feature B (Glucose)", 1, 20, 10)
w_c = st.sidebar.slider("Weight Feature C (RRv)", 1, 20, 15)

P_weights = np.array([w_a, w_b, w_c])

# Computations
composite_risk = np.dot(D_raw, P_weights)
df_risk = pd.DataFrame({'Risk Score': composite_risk}, index=patients)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Composite Risk Scores ($D \\cdot P$)")
    fig_risk = px.bar(df_risk, x=df_risk.index, y='Risk Score', color='Risk Score',
                    color_continuous_scale='Reds', title="Clinical Risk Level per Patient")
    st.plotly_chart(fig_risk, use_container_width=True)

with col2:
    st.subheader("📈 Biomarker Distribution per Patient")
    fig_matrix = px.bar(df_matrix.reset_index(), x='index', y=features, barmode='group',
                        title="Raw Physiological Biomarkers")
    st.plotly_chart(fig_matrix, use_container_width=True)

# Descriptive Statistics
st.subheader("📋 Descriptive Statistics and Variability")
means = np.mean(D_raw, axis=0)
std_devs = np.std(D_raw, axis=0)
cv_percent = (std_devs / means) * 100

df_stats = pd.DataFrame({
    'Mean (μ)': means,
    'Std Dev (σ)': std_devs,
    'CV (%)': cv_percent
}, index=features)

st.dataframe(df_stats.style.highlight_max(axis=0, color='lightgreen'))