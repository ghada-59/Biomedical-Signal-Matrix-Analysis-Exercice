import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Biomedical Feature Matrix Dashboard", page_icon="🧬", layout="wide")

st.title("🧬 Biomedical Signal Matrix & Linear Algebra Dashboard")
st.markdown("Real-time analysis of biomarker variability and composite risk score calculation for intensive care.")

# 1. Raw Data Matrix
D_raw = np.array([
    [4,  8, 12],
    [6, 10, 15],
    [5,  7,  9],
    [3,  6,  9]
])
patients = ['Synthetic sample 1', 'Synthetic sample 2', 'Synthetic sample 3', 'Synthetic sample 4']
features = ['Feature A', 'Feature B', 'Feature C']

df_matrix = pd.DataFrame(D_raw, index=patients, columns=features)

st.sidebar.header("⚙️ Illustrative Weight Configuration")
w_a = st.sidebar.slider("Weight Feature A", 1, 20, 5)
w_b = st.sidebar.slider("Weight Feature B", 1, 20, 10)
w_c = st.sidebar.slider("Weight Feature C", 1, 20, 15)

P_weights = np.array([w_a, w_b, w_c])

# Computations
composite_risk = np.dot(D_raw, P_weights)
df_risk = pd.DataFrame({'Weighted Score': composite_risk}, index=patients)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Composite Weighted Scores ($D \\cdot P$)")
    fig_risk = px.bar(df_risk, x=df_risk.index, y='Weighted Score', color='Weighted Score',
                    color_continuous_scale='Reds', title="Weighted Score Comparison")
    st.plotly_chart(fig_risk, use_container_width=True)

with col2:
    st.subheader("📈 Biomarker Distribution per Patient")
    fig_matrix = px.bar(df_matrix.reset_index(), x='index', y=features, barmode='group',
                        title="Synthetic Feature Values")
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