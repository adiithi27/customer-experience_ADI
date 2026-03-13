# ======================================
# 1. IMPORT LIBRARIES
# ======================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# ======================================
# 2. LOAD DATASET
# ======================================

df = pd.read_csv("cx_dataset.csv")

# View first rows
print(df.head())

# Dataset structure
print(df.info())

# ======================================
# 3. DATA CLEANING
# ======================================

# Remove duplicates
df = df.drop_duplicates()

# Check missing values
print("Missing values:\n", df.isnull().sum())

# Fill missing values with median
df = df.fillna(df.median())

# ======================================
# 4. DESCRIPTIVE STATISTICS
# ======================================

print("\nSummary Statistics")
print(df.describe())

# ======================================
# 5. DISTRIBUTION PLOTS
# ======================================

columns = [
'training_completion_rate',
'onboarding_days',
'support_tickets_per_month',
'user_engagement_score',
'project_mgmt_score',
'first_response_time',
'cx_adoption_success',
'time_to_value',
'customer_retention',
'cxi_score'
]

for col in columns:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# ======================================
# 6. BOXPLOT (OUTLIER DETECTION)
# ======================================

plt.figure(figsize=(12,6))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.title("Outlier Detection Across CX Variables")
plt.show()

# ======================================
# 7. CORRELATION ANALYSIS
# ======================================

corr_matrix = df.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of CX Variables")
plt.show()

# ======================================
# 8. KEY RELATIONSHIP VISUALIZATIONS
# ======================================

# Engagement vs Retention
sns.scatterplot(data=df, x="user_engagement_score", y="customer_retention")
plt.title("User Engagement vs Customer Retention")
plt.show()

# Support Tickets vs CXI Score
sns.scatterplot(data=df, x="support_tickets_per_month", y="cxi_score")
plt.title("Support Tickets vs CXI Score")
plt.show()

# Response Time vs CX Adoption
sns.scatterplot(data=df, x="first_response_time", y="cx_adoption_success")
plt.title("First Response Time vs CX Adoption Success")
plt.show()

# ======================================
# 9. TOP CORRELATIONS
# ======================================

corr_pairs = corr_matrix.unstack().sort_values(ascending=False)

# Remove self correlations
corr_pairs = corr_pairs[corr_pairs != 1]

print("\nTop Positive Correlations:")
print(corr_pairs.head(10))

print("\nTop Negative Correlations:")
print(corr_pairs.tail(10))
