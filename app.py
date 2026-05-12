import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import *

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="SVM Cancer App", layout="wide")

# =========================
# LOAD DATA FUNCTION
# =========================
def load_data(file):
    if file:
        df = pd.read_csv(file)
    else:
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer(as_frame=True)
        df = data.frame
        df.rename(columns={'target': 'diagnosis'}, inplace=True)
    return df

# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess(df):
    if df['diagnosis'].dtype == 'object':
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    for col in ['id', 'Unnamed: 32']:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    return df

# =========================
# TRAIN MODEL FUNCTION
# =========================
def train_model(X, y, kernel, C, gamma, test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
    model.fit(X_train, y_train)

    return model, scaler, X_test, y_test

# =========================
# UI HEADER
# =========================
st.title("🧬 Breast Cancer Prediction using SVM")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Upload Dataset")
file = st.sidebar.file_uploader("CSV File", type=['csv'])

st.sidebar.header("Model Settings")
kernel = st.sidebar.selectbox("Kernel", ['linear', 'rbf', 'poly', 'sigmoid'])
C = st.sidebar.slider("C", 0.01, 100.0, 1.0)
gamma = st.sidebar.selectbox("Gamma", ['scale', 'auto'])
test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)

# =========================
# MAIN DATA
# =========================
df = preprocess(load_data(file))

st.subheader("Dataset Preview")
st.dataframe(df.head())

# =========================
# STATS
# =========================
st.subheader("Dataset Info")

col1, col2 = st.columns(2)
col1.write(df.shape)
col2.write(df.isnull().sum().sum())

# =========================
# VISUALIZATION
# =========================
st.subheader("Class Distribution")
fig = px.histogram(df, x='diagnosis', color='diagnosis')
st.plotly_chart(fig)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(df.corr(), ax=ax, cmap='coolwarm')
st.pyplot(fig)

# =========================
# SPLIT DATA
# =========================
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# =========================
# TRAIN BUTTON
# =========================
if st.button("Train Model"):

    with st.spinner("Training..."):
        model, scaler, X_test, y_test = train_model(
            X, y, kernel, C, gamma, test_size
        )

    X_test_scaled = X_test
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    st.success("Model Trained Successfully")

    # Metrics
    st.subheader("Performance")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred))
    st.write("Recall:", recall_score(y_test, y_pred))
    st.write("F1 Score:", f1_score(y_test, y_pred))

    # Save
    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.X = X

# =========================
# PREDICTION
# =========================
st.subheader("New Prediction")

if 'model' in st.session_state:

    model = st.session_state.model
    scaler = st.session_state.scaler
    X = st.session_state.X

    input_data = {}

    for col in X.columns[:10]:
        input_data[col] = st.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))

    if st.button("Predict"):

        df_input = pd.DataFrame([input_data])

        for col in X.columns:
            if col not in df_input:
                df_input[col] = X[col].mean()

        df_input = df_input[X.columns]

        pred = model.predict(df_input)[0]
        prob = model.predict_proba(df_input)[0][1]

        if pred == 1:
            st.error("Malignant")
        else:
            st.success("Benign")

        st.progress(prob)

else:
    st.warning("Train model first")