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
