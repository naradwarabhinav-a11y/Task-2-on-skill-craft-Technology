# ================================
# TITANIC DATA CLEANING & EDA
# ================================

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load dataset
df = pd.read_csv("train.csv")

# 3. Basic understanding
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

# 4. Check missing values
print(df.isnull().sum())

# 5. Data Cleaning
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)

# Verify cleaning
print(df.isnull().sum())

# ================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ================================

# Survival Count
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# Survival by Gender
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()

# Survival by Passenger Class
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Passenger Class")
plt.show()

# Age Distribution
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# Fare vs Survival
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title("Fare vs Survival")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ================================
# INSIGHTS
# ================================
print("""
INSIGHTS:
1. Females survived more than males
2. First class passengers had higher survival rate
3. Younger passengers survived more
4. Higher fare increased survival probability
""")