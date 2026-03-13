import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="CX Intelligence Dashboard", layout="wide")

sns.set_style("whitegrid")

# Load dataset
df = pd.read_csv("cx_simulated_dataset_400.csv")

st.title("📊 Customer Experience Intelligence Dashboard")

# -----------------------------
# KPI METRICS
# -----------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("CXI Score", f"{df['cxi_score'].mean():.2f}")
col2.metric("Retention", f"{df['customer_retention'].mean():.2f}")
col3.metric("Engagement", f"{df['user_engagement_score'].mean():.2f}")
col4.metric("CX Adoption", f"{df['cx_adoption_success'].mean():.2f}")

st.divider()

# -----------------------------
# FIRST ROW OF CHARTS
# -----------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("CXI Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["cxi_score"], kde=True, ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Customer Retention Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["customer_retention"], kde=True, ax=ax)
    st.pyplot(fig)

# -----------------------------
# SECOND ROW OF CHARTS
# -----------------------------

col3, col4 = st.columns(2)

with col3:
    st.subheader("Engagement vs Retention")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="user_engagement_score", y="customer_retention", ax=ax)
    st.pyplot(fig)

with col4:
    st.subheader("Support Tickets vs CXI Score")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="support_tickets_per_month", y="cxi_score", ax=ax)
    st.pyplot(fig)

# -----------------------------
# THIRD ROW
# -----------------------------

col5, col6 = st.columns(2)

with col5:
    st.subheader("Response Time vs CX Adoption")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="first_response_time", y="cx_adoption_success", ax=ax)
    st.pyplot(fig)

with col6:
    st.subheader("Training Completion vs CXI")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="training_completion_rate", y="cxi_score", ax=ax)
    st.pyplot(fig)

# -----------------------------
# CORRELATION HEATMAP
# -----------------------------

st.subheader("Correlation Between CX Metrics")

fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)

st.pyplot(fig)

# -----------------------------
# DATA PREVIEW
# -----------------------------

with st.expander("View Dataset"):
    st.dataframe(df)
