import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.title("Exploratory Data Analysis (EDA)")

@st.cache_data
def load_data():
    # CORRECT PATH (same as your project folder)
    csv_path = os.path.join("data", "raw", "Churn_Modelling.csv")
    df = pd.read_csv(csv_path)
    return df

df = load_data()

st.subheader("Dataset Preview")
st.write(df.head())

st.subheader("Summary Statistics")
st.write(df.describe())

# Only numeric columns for heatmap
numeric_df = df.select_dtypes(include=['int64', 'float64'])

st.subheader("Correlation Heatmap")

plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
st.pyplot(plt)
plt.clf()
st.subheader("Churn Distribution")

fig, ax = plt.subplots()
df['Exited'].value_counts().plot(kind='bar', ax=ax)

ax.set_xlabel("Exited")
ax.set_ylabel("Count")

st.pyplot(fig)
st.subheader("Geography-wise Churn Rate")

geo_churn = df.groupby('Geography')['Exited'].mean() * 100

fig, ax = plt.subplots()
geo_churn.plot(kind='bar', ax=ax)

ax.set_ylabel("Churn Rate (%)")

st.pyplot(fig)
st.subheader("Age Distribution")

fig, ax = plt.subplots()
df['Age'].hist(ax=ax, bins=20)

ax.set_xlabel("Age")
ax.set_ylabel("Customers")

st.pyplot(fig)