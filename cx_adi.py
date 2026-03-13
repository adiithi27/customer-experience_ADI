import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="CX Intelligence Dashboard", layout="wide")

st.title("Customer Experience Intelligence Dashboard")

# Load dataset
df = pd.read_csv("cx_simulated_dataset_400.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ==============================
# KPI METRICS
# ==============================

st.subheader("Key CX Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Average CXI Score", round(df["cxi_score"].mean(),2))
col2.metric("Customer Retention", round(df["customer_retention"].mean(),2))
col3.metric("User Engagement Score", round(df["user_engagement_score"].mean(),2))
col4.metric("CX Adoption Success", round(df["cx_adoption_success"].mean(),2))

# ==============================
# DESCRIPTIVE STATISTICS
# ==============================

st.subheader("Descriptive Statistics")
st.write(df.describe())

# ==============================
# DISTRIBUTION ANALYSIS
# ==============================

st.subheader("Variable Distribution")

variable = st.selectbox(
    "Select Variable",
    df.columns
)

fig, ax = plt.subplots()

sns.histplot(df[variable], kde=True, ax=ax)

ax.set_title(f"Distribution of {variable}")

st.pyplot(fig)

# ==============================
# CORRELATION HEATMAP
# ==============================

st.subheader("Correlation Heatmap")

corr = df.corr()

fig2, ax2 = plt.subplots(figsize=(10,6))

sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)

st.pyplot(fig2)

# ==============================
# RELATIONSHIP ANALYSIS
# ==============================

st.subheader("CX Relationship Analysis")

x_var = st.selectbox("Select X Variable", df.columns)
y_var = st.selectbox("Select Y Variable", df.columns)

fig3, ax3 = plt.subplots()

sns.scatterplot(data=df, x=x_var, y=y_var, ax=ax3)

ax3.set_title(f"{x_var} vs {y_var}")

st.pyplot(fig3)
