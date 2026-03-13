import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="CX Intelligence Dashboard", layout="wide")

# Load dataset
df = pd.read_csv("cx_simulated_dataset_400.csv")

st.title("📊 Customer Experience Intelligence Dashboard")

# --------------------------------------------------
# KPI METRICS
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("CXI Score", f"{df['cxi_score'].mean():.2f}")
col2.metric("Customer Retention", f"{df['customer_retention'].mean():.2f}")
col3.metric("User Engagement", f"{df['user_engagement_score'].mean():.2f}")
col4.metric("CX Adoption", f"{df['cx_adoption_success'].mean():.2f}")

st.divider()

# --------------------------------------------------
# CUSTOMER HEALTH GAUGE
# --------------------------------------------------

st.subheader("Customer Experience Health Score")

health_score = df["cxi_score"].mean() * 10

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=health_score,
    title={'text': "CX Health Score"},
    gauge={
        'axis': {'range': [0,100]},
        'bar': {'color': "green"},
        'steps': [
            {'range':[0,40], 'color':"red"},
            {'range':[40,70], 'color':"orange"},
            {'range':[70,100], 'color':"lightgreen"}
        ]
    }
))

st.plotly_chart(fig, use_container_width=True)

st.divider()

# --------------------------------------------------
# MULTI GRAPH ANALYTICS
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    fig = px.scatter(
        df,
        x="user_engagement_score",
        y="customer_retention",
        title="Engagement vs Retention"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:

    fig = px.scatter(
        df,
        x="support_tickets_per_month",
        y="cxi_score",
        title="Support Tickets vs CXI Score"
    )

    st.plotly_chart(fig, use_container_width=True)

col3, col4 = st.columns(2)

with col3:

    fig = px.histogram(
        df,
        x="cxi_score",
        title="CXI Score Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

with col4:

    fig = px.histogram(
        df,
        x="customer_retention",
        title="Retention Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --------------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------------

st.subheader("Correlation Between CX Metrics")

fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)

st.pyplot(fig)

st.divider()

# --------------------------------------------------
# DRIVER ANALYSIS (Random Forest)
# --------------------------------------------------

st.subheader("Key Drivers of CXI Score")

X = df.drop(columns=["cxi_score"])
y = df["cxi_score"]

model = RandomForestRegressor()
model.fit(X,y)

importance = pd.DataFrame({
    "Feature":X.columns,
    "Importance":model.feature_importances_
}).sort_values(by="Importance", ascending=False)

fig = px.bar(
    importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Drivers of CXI Score"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# --------------------------------------------------
# AUTOMATED CX INSIGHTS
# --------------------------------------------------

st.subheader("Automated CX Insights")

top_driver = importance.iloc[0]["Feature"]

st.write(f"🔎 **Top CX Driver:** {top_driver}")

if df["user_engagement_score"].corr(df["customer_retention"]) > 0.5:
    st.write("📈 Higher engagement strongly correlates with customer retention.")

if df["support_tickets_per_month"].corr(df["cxi_score"]) < 0:
    st.write("⚠️ More support tickets are associated with lower CXI scores.")

if df["training_completion_rate"].corr(df["cx_adoption_success"]) > 0.4:
    st.write("🎓 Training completion improves CX adoption success.")

st.divider()

with st.expander("View Dataset"):
    st.dataframe(df)
