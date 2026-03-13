import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="CX Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

sns.set_style("whitegrid")

# -----------------------------------
# LOAD DATA
# -----------------------------------

df = pd.read_csv("cx_simulated_dataset_400.csv")

# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title("CX Dashboard Controls")

selected_variable = st.sidebar.selectbox(
    "Select variable for distribution",
    df.columns
)

x_var = st.sidebar.selectbox("X variable", df.columns)
y_var = st.sidebar.selectbox("Y variable", df.columns)

# -----------------------------------
# HEADER
# -----------------------------------

st.title("📊 Customer Experience Intelligence Dashboard")
st.markdown("Monitor key drivers of **customer experience, adoption, and retention**.")

st.divider()

# -----------------------------------
# KPI SECTION
# -----------------------------------

st.subheader("Key CX Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("CXI Score", f"{df['cxi_score'].mean():.2f}")
col2.metric("Customer Retention", f"{df['customer_retention'].mean():.2f}")
col3.metric("User Engagement", f"{df['user_engagement_score'].mean():.2f}")
col4.metric("CX Adoption Success", f"{df['cx_adoption_success'].mean():.2f}")

st.divider()

# -----------------------------------
# DATA PREVIEW
# -----------------------------------

with st.expander("View Dataset"):
    st.dataframe(df)

# -----------------------------------
# DISTRIBUTION ANALYSIS
# -----------------------------------

st.subheader("Variable Distribution")

fig, ax = plt.subplots()

sns.histplot(df[selected_variable], kde=True, ax=ax)

ax.set_title(f"Distribution of {selected_variable}")

st.pyplot(fig)

st.divider()

# -----------------------------------
# CORRELATION HEATMAP
# -----------------------------------

st.subheader("Correlation Analysis")

corr = df.corr()

fig2, ax2 = plt.subplots(figsize=(10,6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5,
    ax=ax2
)

ax2.set_title("CX Metrics Correlation Heatmap")

st.pyplot(fig2)

st.divider()

# -----------------------------------
# RELATIONSHIP ANALYSIS
# -----------------------------------

st.subheader("CX Relationship Analysis")

fig3, ax3 = plt.subplots()

sns.scatterplot(
    data=df,
    x=x_var,
    y=y_var,
    s=80,
    alpha=0.7,
    ax=ax3
)

ax3.set_title(f"{x_var} vs {y_var}")

st.pyplot(fig3)

st.divider()

# -----------------------------------
# DESCRIPTIVE STATS
# -----------------------------------

st.subheader("Descriptive Statistics")

st.dataframe(df.describe())
