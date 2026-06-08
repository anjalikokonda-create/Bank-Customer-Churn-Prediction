import streamlit as st
import pandas as pd

st.title("Bank Customer Churn Analysis")

# Load CSV
df = pd.read_csv("data/raw/Churn_Modelling.csv")

st.subheader("Raw Dataset")
st.write(df)

st.subheader("Show Code")
st.code("""
import pandas as pd
df = pd.read_csv("data/raw/Churn_Modelling.csv")
""")