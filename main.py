import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="Biomedical Feature Matrix & Linear Algebra Analysis",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Synthetic Biomedical Feature Matrix Analysis")
st.markdown("""
**Domain:** Biomedical Data & AI Engineering  
This interactive dashboard uses a small synthetic feature matrix to explore statistics, matrix operations and weighted linear combinations.
""")

st.divider()

# ------------------------------------------------------------------------------
# 1. DATASET INITIALIZATION
# ------------------------------------------------------------------------------
D_raw = np.array([
    [4,  8, 12],  # Patient 1
    [6, 10, 15],  # Patient 2
    [5,  7,  9],  # Patient 3
    [3,  6,  9]   # Patient 4
])

samples = ['Sample_1', 'Sample_2', 'Sample_3', 'Sample_4']
features = ['Feature_A (HRV)', 'Feature_B (Glucose)', 'Feature_C (RRv)']

# Sidebar for Dynamic Diagnostic Weighting
st.sidebar.header("⚙️ Feature Weights (P)")
st.sidebar.markdown("Adjust the weights applied to each illustrative feature:")
w_a = st.sidebar.slider("Weight Feature A (HRV)", 1, 20, 5)
w_b = st.sidebar.slider("Weight Feature B (Glucose)", 1, 20, 10)
w_c = st.sidebar.slider("Weight Feature C (RRv)", 1, 20, 15)

P_weights = np.array([w_a, w_b, w_c])

# ------------------------------------------------------------------------------
# COMPUTATIONS
# ------------------------------------------------------------------------------
# Part 1: Descriptive Statistics
means = np.mean(D_raw, axis=0)
medians = np.median(D_raw, axis=0)
std_devs = np.std(D_raw, axis=0, ddof=0)
ranges = np.max(D_raw, axis=0) - np.min(D_raw, axis=0)
variances = np.var(D_raw, axis=0, ddof=0)
cv_percentages = (std_devs / means) * 100

df_stats = pd.DataFrame({
    'Mean (μ)': means,
    'Median': medians,
    'Std Dev (σ)': std_devs,
    'Range': ranges,
    'Variance (σ²)': variances,
    'CV (%)': cv_percentages
}, index=features)

# Part 2: Linear Algebra Operations
D_transpose = D_raw.T
composite_risk_scores = np.dot(D_raw, P_weights)

df_risk = pd.DataFrame({'Composite Risk Score': composite_risk_scores}, index=samples)
highest_score_sample = df_risk.idxmax().iloc[0]
max_score = df_risk.max().iloc[0]

total_biomarker_severity = np.sum(D_raw * P_weights, axis=0)
df_severity = pd.DataFrame({'Total Severity Contribution': total_biomarker_severity}, index=features)

# ------------------------------------------------------------------------------
# LAYOUT & DISPLAY
# ------------------------------------------------------------------------------

# Summary
st.warning(f"🚨 **CRITICAL ALERT:** Highest Risk Patient identified as **{highest_risk_patient}** with a score of **{max_score} points**.")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 1. Synthetic Feature Matrix (D)")
    df_biomed = pd.DataFrame(D_raw, index=patients, columns=features)
    st.dataframe(df_biomed, use_container_width=True)
    st.caption(f"Matrix Elements Data Type: `{D_raw.dtype}` (Integer-encoded ratio scale data)")

    st.subheader("🎯 2. Weighted Composite Scores (R = D · P)")
    fig_risk = px.bar(
        df_risk.reset_index(),
        x='index',
        y='Composite Risk Score',
        color='Composite Risk Score',
        color_continuous_scale='Reds',
        labels={'index': 'Patient', 'Composite Risk Score': 'Risk Score'},
        title="Weighted Score Comparison"
    )
    st.plotly_chart(fig_risk, use_container_width=True)

with col_right:
    st.subheader("🔄 3. Transposed Matrix (Dᵀ)")
    df_transpose = pd.DataFrame(D_transpose, index=['HRV (A)', 'Glucose (B)', 'RRv (C)'], columns=samples)
    st.dataframe(df_transpose, use_container_width=True)

    st.subheader("📈 4. Total Biomarker Cohort Contribution")
    fig_severity = px.bar(
        df_severity.reset_index(),
        x='index',
        y='Total Severity Contribution',
        color='index',
        labels={'index': 'Biomarker Feature'},
        title="Aggregate Severity per Feature Across Cohort"
    )
    st.plotly_chart(fig_severity, use_container_width=True)

st.divider()

# Part 3: Descriptive Statistics Table
st.subheader("📋 Descriptive Statistics & Signal Variability")
st.dataframe(df_stats.round(2).style.highlight_min(subset=['CV (%)'], color='lightgreen').highlight_max(subset=['CV (%)'], color='pink'), use_container_width=True)

st.divider()

# Part 4: Interpretation & AI Engineering Context
st.subheader("💡 Interpretation & AI/Engineering Context")

tab1, tab2 = st.tabs(["📊 Matrix Interpretation", "🤖 AI/Engineering Context"])

with tab1:
    st.markdown("""
    * **Mean ($\mu$):** Represents baseline physiological signal intensity across the cohort. Feature C exhibits the highest intensity ($\mu = 11.25$).
    * **Coefficient of Variation ($CV\%$):** Standardizes relative dispersion. **Feature B (Glucose)** ($CV = 19.08\%$) is the most stable biomarker across patients, whereas **Feature A (HRV)** ($CV = 24.85\%$) shows the highest relative inter-patient dispersion.
    * **Patient Triage:** **Patient 2** requires immediate clinical intervention with a dominant composite risk score of **355 points**.
    """)

with tab2:
    st.markdown("""
    1. **Neural Network Layers (1D-CNNs / MLPs):** Dot product transformations ($\mathbf{W}^T \mathbf{X} + \mathbf{b}$) are the fundamental operation in deep learning architectures used to process 1D physiological signals for continuous arrhythmia detection.
    2. **Dimensionality Reduction (PCA / SVD):** Singular Value Decomposition ($D = U \Sigma V^T$) compresses dynamic biomarker matrices to reduce high-dimensional patient data into informative feature spaces without significant signal loss.
    """)