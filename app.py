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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(
    page_title="SVM Cancer Diagnosis App",
    page_icon="🧬",
    layout="wide"
)

# ==========================================
# SMALL UI IMPROVEMENT CSS
# ==========================================
st.markdown("""
<style>
.main {background-color: #f7f9fc;}

.metric-box {
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    font-weight: 600;
    color: white;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.15);
}

.green {background: linear-gradient(135deg,#11998e,#38ef7d);}
.red {background: linear-gradient(135deg,#ff416c,#ff4b2b);}
.blue {background: linear-gradient(135deg,#396afc,#2948ff);}
.orange {background: linear-gradient(135deg,#f7971e,#ffd200);}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================
st.title("🧬 SVM Cancer Diagnosis System")
st.caption("Interactive ML Dashboard for Breast Cancer Prediction")

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("📁 Dataset Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# SVM SETTINGS
st.sidebar.markdown("## ⚙️ Model Settings")
kernel = st.sidebar.selectbox("Kernel Type", ['linear', 'rbf', 'poly', 'sigmoid'])
C = st.sidebar.slider("Regularization (C)", 0.01, 100.0, 1.0)
gamma = st.sidebar.selectbox("Gamma", ['scale', 'auto'])
test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)

# ==========================================
# LOAD DATA
# ==========================================
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    df.rename(columns={'target': 'diagnosis'}, inplace=True)

# Convert labels if needed
if df['diagnosis'].dtype == 'object':
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Drop unnecessary columns
df.drop(columns=[c for c in ['id', 'Unnamed: 32'] if c in df.columns], inplace=True)

# ==========================================
# DATA PREVIEW
# ==========================================
st.markdown("## 📊 Dataset Overview")
st.dataframe(df.head(), use_container_width=True)

# ==========================================
# METRICS
# ==========================================
st.markdown("## 📌 Dataset Summary")

rows, cols = df.shape
malignant = (df['diagnosis'] == 1).sum()
benign = (df['diagnosis'] == 0).sum()

c1, c2, c3, c4 = st.columns(4)

c1.markdown(f"<div class='metric-box blue'>Rows<br>{rows}</div>", unsafe_allow_html=True)
c2.markdown(f"<div class='metric-box orange'>Columns<br>{cols}</div>", unsafe_allow_html=True)
c3.markdown(f"<div class='metric-box red'>Malignant<br>{malignant}</div>", unsafe_allow_html=True)
c4.markdown(f"<div class='metric-box green'>Benign<br>{benign}</div>", unsafe_allow_html=True)

# ==========================================
# DISTRIBUTION
# ==========================================
st.markdown("## ⚖️ Class Distribution")

fig = px.histogram(df, x='diagnosis', color='diagnosis', text_auto=True)
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# HEATMAP
# ==========================================
st.markdown("## 🔥 Correlation Heatmap")

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ==========================================
# FEATURES
# ==========================================
st.markdown("## 📈 Feature Analysis")

feature = st.selectbox("Select Feature", df.drop('diagnosis', axis=1).columns)

fig = go.Figure()
fig.add_trace(go.Histogram(x=df[df['diagnosis']==0][feature], name="Benign", opacity=0.6))
fig.add_trace(go.Histogram(x=df[df['diagnosis']==1][feature], name="Malignant", opacity=0.6))
fig.update_layout(barmode='overlay', title=f"{feature} Distribution")
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# SPLIT DATA
# ==========================================
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# TRAIN MODEL
# ==========================================
st.markdown("## 🚀 Train Model")

if st.button("Train SVM Model"):

    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.markdown("## 📊 Model Performance")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("Precision", f"{prec:.3f}")
    m3.metric("Recall", f"{rec:.3f}")
    m4.metric("F1 Score", f"{f1:.3f}")

    # Confusion Matrix
    st.markdown("## 🧩 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    # ROC Curve
    st.markdown("## 📉 ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC={roc_auc:.3f}"))
    st.plotly_chart(fig, use_container_width=True)

    st.session_state['model'] = model
    st.session_state['scaler'] = scaler

# ==========================================
# PREDICTION
# ==========================================
st.markdown("## 🔍 Predict New Patient")

if 'model' in st.session_state:

    model = st.session_state['model']
    scaler = st.session_state['scaler']

    input_data = {}

    for col in X.columns[:10]:
        input_data[col] = st.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))

    if st.button("Predict"):

        input_df = pd.DataFrame([input_data])

        for col in X.columns:
            if col not in input_df:
                input_df[col] = X[col].mean()

        input_df = input_df[X.columns]
        input_scaled = scaler.transform(input_df)

        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if pred == 1:
            st.error("🔴 Malignant (High Risk)")
        else:
            st.success("🟢 Benign (Low Risk)")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Cancer Probability"},
            gauge={'axis': {'range': [0, 100]}}
        ))

        st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Train the model first ⚠️")