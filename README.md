# 🧬 Biomedical Feature Matrix & Linear Algebra Exercise

A small academic exercise exploring **matrix operations, descriptive statistics and weighted linear combinations** with a synthetic biomedical-style feature matrix.

## 🎯 Objective

The project uses a **4 × 3 synthetic matrix** to practice descriptive statistics, matrix transpose, vector dot products, weighted feature combinations and simple visualization.

## 🔬 Example Matrix

```text
D =
[ 4   8  12 ]
[ 6  10  15 ]
[ 5   7   9 ]
[ 3   6   9 ]
```

A weight vector is applied with:

```text
R = D · P
```

The resulting values are interpreted only as **weighted scores for the exercise**. They are not normalized and are not clinical measurements.

## 📊 Analysis

The dashboard computes the statistics directly from the fixed 4 × 3 matrix and recomputes the weighted score whenever the illustrative weights are changed. The default weights are `[5, 10, 15]`.

- mean, median, standard deviation, range and variance
- coefficient of variation
- matrix transpose
- weighted composite scores
- feature-level contribution to the weighted score
- simple visualizations

## 🛠️ Technologies

Python · NumPy · Pandas · Plotly · Streamlit

## ⚠️ Scope and limitations

This repository uses a **synthetic toy dataset**. The features and weights are illustrative and are not validated physiological measurements, clinical risk scores, diagnostic models, or patient-triage tools.

The exercise demonstrates mathematical foundations that can later be used in data-science and machine-learning workflows.

### Validation

The implementation keeps the matrix dimensions consistent (`4 × 3`), uses a length-3 weight vector for the dot product, and derives the displayed statistics and weighted contributions directly from the same synthetic matrix. The weighted score is recalculated from the user-selected weights on each Streamlit rerun.

Because the dataset is intentionally tiny and synthetic, model-performance metrics are not appropriate here. Validation focuses on mathematical consistency and transparent computation rather than predictive performance.

## 🚀 Run

```bash
pip install -r requirements.txt
streamlit run main.py
```
