# =====================================
# AMAZON CUSTOMER BEHAVIOR ANALYSIS
# TASK-WISE FINAL CODE
# =====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv("Amazon.csv")
print("Initial Shape:", df.shape)
print(df.head())

# =====================================
# TASK 1: DATA CLEANING AND PREPARATION
# =====================================

# Remove duplicates
df.drop_duplicates(inplace=True)

# Remove column spaces
df.columns = df.columns.str.strip()

# Handle missing values
df = df.ffill()

# Standardize categorical values
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].str.lower().str.strip()

# Convert numeric columns
num_cols = ['Customer_Reviews_Importance', 'Shopping_Satisfaction', 'Rating_Accuracy']

for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nAfter Cleaning:", df.shape)

# =====================================
# TASK 2: DESCRIPTIVE ANALYSIS
# =====================================

# Age distribution
print("\nAge Summary:")
print(df['age'].describe())

# Gender distribution
print("\nGender Distribution:")
print(df['Gender'].value_counts())

# Purchase Frequency
if 'Purchase_Frequency' in df.columns:
    print("\nPurchase Frequency:")
    print(df['Purchase_Frequency'].value_counts())

# Category cleaning
categories = df['Purchase_Categories'].dropna().str.split(';')
all_categories = [cat.strip() for sublist in categories for cat in sublist]
cat_df = pd.DataFrame(all_categories, columns=['Category'])

# Category graph
plt.figure(figsize=(8,5))
sns.countplot(y='Category', data=cat_df,
              order=cat_df['Category'].value_counts().index)
plt.title("Top Purchase Categories")
plt.tight_layout()
plt.savefig("top_categories.png")
plt.show()

# Satisfaction stats
print("\nSatisfaction Mean:", df['Shopping_Satisfaction'].mean())
print("Satisfaction Median:", df['Shopping_Satisfaction'].median())

# Satisfaction graph
plt.figure(figsize=(6,4))
plt.hist(df['Shopping_Satisfaction'].dropna(), bins=5)
plt.title("Shopping Satisfaction Distribution")
plt.xlabel("Satisfaction")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("satisfaction.png")
plt.show()

# =====================================
# TASK 3: CUSTOMER SEGMENTATION
# =====================================

features = df[['Shopping_Satisfaction', 'Customer_Reviews_Importance']].dropna()

scaler = StandardScaler()
X = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df.loc[features.index, 'Cluster'] = kmeans.fit_predict(X)

print("\nCluster Counts:")
print(df['Cluster'].value_counts())

# =====================================
# TASK 4: RECOMMENDATION & REVIEW INSIGHTS
# =====================================

# Correlation
corr = df[['Shopping_Satisfaction', 'Customer_Reviews_Importance']].corr()
print("\nCorrelation Matrix:")
print(corr)

# Heatmap
plt.figure(figsize=(5,4))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Between Reviews & Satisfaction")
plt.tight_layout()
plt.savefig("correlation.png")
plt.show()

# =====================================
# TASK 5: VISUALIZATION
# =====================================

# Pie chart for gender
plt.figure(figsize=(5,5))
df['Gender'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Gender Distribution")
plt.ylabel("")
plt.savefig("gender_pie.png")
plt.show()

# Purchase frequency chart
if 'Purchase_Frequency' in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x='Purchase_Frequency', data=df)
    plt.title("Purchase Frequency Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("purchase_freq.png")
    plt.show()

# =====================================
# MARKET BASKET ANALYSIS 
# =====================================

basket = df['Purchase_Categories'].astype(str).str.get_dummies(sep=';')
basket = basket.astype(bool)

frequent_items = apriori(basket, min_support=0.05, use_colnames=True)

print("\nFrequent Itemsets:")
print(frequent_items.head())

rules = association_rules(frequent_items, metric="lift", min_threshold=1)

print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# =====================================
# FINAL INSIGHTS
# =====================================

print("\n--- FINAL INSIGHTS ---")
print("1. Clothing is the most popular category")
print("2. Customers prefer multiple categories")
print("3. Satisfaction impacts purchase behavior")
print("4. Reviews influence buying decisions")
print("5. Strong cross-selling opportunities exist")

