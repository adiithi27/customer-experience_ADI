# ===============================
# 1. IMPORT LIBRARIES
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ===============================
# 2. LOAD DATASET
# ===============================

# Replace with your file path
df = pd.read_excel("cx_dataset.xlsx")

# View dataset
print(df.head())
print(df.info())

# ===============================
# 3. DATA CLEANING
# ===============================

# Remove duplicate rows
df = df.drop_duplicates()

# Check missing values
print(df.isnull().sum())

# Fill numeric missing values with median
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill categorical missing values with mode
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# ===============================
# 4. DATA TRANSFORMATION
# ===============================

# Example transformations (modify based on your variables)

# Convert date column
# df['interaction_date'] = pd.to_datetime(df['interaction_date'])

# Create derived features
# Example: Customer tenure
# df['customer_tenure_days'] = (pd.Timestamp.today() - df['signup_date']).dt.days

# Encoding categorical variables (if needed)
df_encoded = pd.get_dummies(df, drop_first=True)

# ===============================
# 5. DESCRIPTIVE ANALYTICS
# ===============================

print("\nDescriptive Statistics")
print(df.describe())

print("\nCategorical Distribution")
for col in categorical_cols:
    print(df[col].value_counts())

# ===============================
# 6. EDA VISUALIZATIONS
# ===============================

# Distribution plots for numerical variables
df[numeric_cols].hist(figsize=(15,10))
plt.suptitle("Distribution of Numeric Variables")
plt.show()

# Boxplots for outlier detection
plt.figure(figsize=(15,6))
sns.boxplot(data=df[numeric_cols])
plt.title("Outlier Detection")
plt.xticks(rotation=45)
plt.show()

# Count plots for categorical variables
for col in categorical_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x=col)
    plt.xticks(rotation=45)
    plt.title(f"Distribution of {col}")
    plt.show()

# ===============================
# 7. CORRELATION ANALYSIS
# ===============================

corr_matrix = df_encoded.corr()

plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.show()

# ===============================
# 8. TOP CORRELATIONS
# ===============================

corr_pairs = corr_matrix.unstack().sort_values(kind="quicksort", ascending=False)

# Remove duplicate correlations
corr_pairs = corr_pairs[corr_pairs != 1]

print("Top Positive Correlations")
print(corr_pairs.head(10))

print("Top Negative Correlations")
print(corr_pairs.tail(10))

# ===============================
# 9. KEY CX VISUALIZATIONS
# ===============================

# Example: Customer Satisfaction vs Response Time

if 'customer_satisfaction_score' in df.columns and 'response_time' in df.columns:
    plt.figure(figsize=(7,5))
    sns.scatterplot(data=df, x='response_time', y='customer_satisfaction_score')
    plt.title("Response Time vs Customer Satisfaction")
    plt.show()

# Example: Channel performance
if 'interaction_channel' in df.columns and 'customer_satisfaction_score' in df.columns:
    plt.figure(figsize=(7,5))
    sns.boxplot(data=df, x='interaction_channel', y='customer_satisfaction_score')
    plt.title("Customer Satisfaction by Channel")
    plt.xticks(rotation=45)
    plt.show()

# ===============================
# 10. SAVE CLEANED DATASET
# ===============================

df.to_excel("cleaned_cx_dataset.xlsx", index=False)

print("Data cleaning and EDA completed successfully!")
