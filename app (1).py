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
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="SVM Cancer Diagnosis Application",
    page_icon="🧬",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.metric-card {
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
}

.green {
    background: linear-gradient(135deg, #00b09b, #96c93d);
}

.yellow {
    background: linear-gradient(135deg, #f7971e, #ffd200);
}

.red {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================

st.title("🧬 SVM Cancer Diagnosis Application")
st.markdown("## Interactive Machine Learning Web App")

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=['csv']
)

# ==========================================
# LOAD DATA
# ==========================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

else:
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True)

    df = data.frame

    df.rename(columns={'target':'diagnosis'}, inplace=True)

# ==========================================
# HANDLE DIAGNOSIS COLUMN
# ==========================================

if df['diagnosis'].dtype == 'object':

    if 'M' in df['diagnosis'].unique():

        df['diagnosis'] = df['diagnosis'].map({
            'M':1,
            'B':0
        })

# ==========================================
# DROP UNWANTED COLUMNS
# ==========================================

drop_cols = ['id', 'Unnamed: 32']

for col in drop_cols:

    if col in df.columns:
        df.drop(columns=col, inplace=True)

# ==========================================
# DATASET OVERVIEW
# ==========================================

st.markdown("---")

st.subheader("📊 Dataset Preview")

st.dataframe(df.head(), use_container_width=True)

# ==========================================
# METRICS
# ==========================================

rows = df.shape[0]
cols = df.shape[1]
missing = df.isnull().sum().sum()

malignant = (df['diagnosis'] == 1).sum()
benign = (df['diagnosis'] == 0).sum()

m_percent = (malignant / len(df)) * 100
b_percent = (benign / len(df)) * 100

c1, c2, c3, c4 = st.columns(4)

c1.markdown(f"""
<div class='metric-card green'>
Rows<br>{rows}
</div>
""", unsafe_allow_html=True)

c2.markdown(f"""
<div class='metric-card yellow'>
Columns<br>{cols}
</div>
""", unsafe_allow_html=True)

c3.markdown(f"""
<div class='metric-card red'>
Malignant %<br>{m_percent:.2f}
</div>
""", unsafe_allow_html=True)

c4.markdown(f"""
<div class='metric-card green'>
Benign %<br>{b_percent:.2f}
</div>
""", unsafe_allow_html=True)

# ==========================================
# NULL VALUES
# ==========================================

st.markdown("---")

st.subheader("🕳️ Missing Values")

st.write(df.isnull().sum())

# ==========================================
# CLASS DISTRIBUTION
# ==========================================

st.markdown("---")

st.subheader("⚖️ Diagnosis Distribution")

counts = df['diagnosis'].value_counts().reset_index()

counts.columns = ['Diagnosis', 'Count']

fig = px.bar(
    counts,
    x='Diagnosis',
    y='Count',
    color='Diagnosis',
    text='Count',
    title='Benign vs Malignant'
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# CORRELATION HEATMAP
# ==========================================

st.markdown("---")

st.subheader("🔥 Correlation Heatmap")

fig, ax = plt.subplots(figsize=(18,14))

sns.heatmap(
    df.corr(),
    cmap='coolwarm',
    ax=ax
)

st.pyplot(fig)

# ==========================================
# FEATURE DISTRIBUTION
# ==========================================

st.markdown("---")

st.subheader("📈 Feature Distribution")

feature = st.selectbox(
    "Select Feature",
    df.drop('diagnosis', axis=1).columns
)

fig = go.Figure()

fig.add_trace(
    go.Histogram(
        x=df[df['diagnosis']==0][feature],
        name='Benign',
        opacity=0.6
    )
)

fig.add_trace(
    go.Histogram(
        x=df[df['diagnosis']==1][feature],
        name='Malignant',
        opacity=0.6
    )
)

fig.update_layout(
    barmode='overlay',
    title=f'Distribution of {feature}'
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# FEATURE SELECTION
# ==========================================

X = df.drop('diagnosis', axis=1)

y = df['diagnosis']

# ==========================================
# SIDEBAR SETTINGS
# ==========================================

st.sidebar.markdown("---")

st.sidebar.header("⚙️ SVM Settings")

kernel = st.sidebar.selectbox(
    "Kernel",
    ['linear', 'rbf', 'poly', 'sigmoid']
)

C = st.sidebar.slider(
    "Regularization (C)",
    0.01,
    100.0,
    1.0
)

gamma = st.sidebar.selectbox(
    "Gamma",
    ['scale', 'auto']
)

test_size = st.sidebar.slider(
    "Test Size",
    0.1,
    0.4,
    0.2
)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42,
    stratify=y
)

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# ==========================================
# TRAIN BUTTON
# ==========================================

if st.button("🚀 Train SVM Model"):

    progress = st.progress(0)

    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    # ======================================
    # MODEL
    # ======================================

    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        probability=True
    )

    model.fit(X_train_scaled, y_train)

    # ======================================
    # PREDICTIONS
    # ======================================

    y_pred = model.predict(X_test_scaled)

    y_prob = model.predict_proba(X_test_scaled)[:,1]

    # ======================================
    # METRICS
    # ======================================

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    st.markdown("---")

    st.subheader("📈 Model Performance")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Accuracy", f"{accuracy:.3f}")

    m2.metric("Precision", f"{precision:.3f}")

    m3.metric("Recall", f"{recall:.3f}")

    m4.metric("F1 Score", f"{f1:.3f}")

    # ======================================
    # CONFUSION MATRIX
    # ======================================

    st.markdown("---")

    st.subheader("🧩 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    # ======================================
    # CLASSIFICATION REPORT
    # ======================================

    st.markdown("---")

    st.subheader("📋 Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df, use_container_width=True)

    # ======================================
    # ROC CURVE
    # ======================================

    st.markdown("---")

    st.subheader("📉 ROC Curve")

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    roc_auc = auc(fpr, tpr)

    roc_fig = go.Figure()

    roc_fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'AUC = {roc_auc:.3f}'
        )
    )

    roc_fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate'
    )

    st.plotly_chart(roc_fig, use_container_width=True)

    # ======================================
    # SAVE MODEL
    # ======================================

    st.session_state['model'] = model
    st.session_state['scaler'] = scaler

# ==========================================
# PATIENT PREDICTION
# ==========================================

st.markdown("---")

st.subheader("🔍 Predict New Patient")

if 'model' not in st.session_state:

    st.warning("⚠️ Train the model first")

else:

    model = st.session_state['model']

    scaler = st.session_state['scaler']

    input_data = {}

    feature_cols = X.columns[:10]

    for col in feature_cols:

        input_data[col] = st.slider(
            col,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )

    if st.button("🩺 Diagnose"):

        input_df = pd.DataFrame([input_data])

        missing_cols = list(
            set(X.columns) - set(input_df.columns)
        )

        for col in missing_cols:

            input_df[col] = X[col].mean()

        input_df = input_df[X.columns]

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]

        probability = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:

            st.error("🔴 MALIGNANT — High Cancer Risk")

        else:

            st.success("🟢 BENIGN — Low Cancer Risk")

        # ======================================
        # GAUGE CHART
        # ======================================

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability*100,
                title={'text':'Cancer Probability'},
                gauge={
                    'axis': {'range':[0,100]}
                }
            )
        )

        st.plotly_chart(gauge, use_container_width=True)