import streamlit as st
import numpy as np
import joblib

model = joblib.load("models/final_model.joblib")
scaler = joblib.load("models/preprocessor.joblib")

st.title("Bank Customer Churn Prediction")

credit_score = st.number_input("Credit Score", 300, 900)
age = st.number_input("Age", 18, 100)
balance = st.number_input("Balance")
salary = st.number_input("Salary")

if st.button("Predict"):
    data = np.array([[credit_score, 0, 1, age, 5, balance, 1, 1, 1, salary]])
    data = scaler.transform(data)
    result = model.predict(data)[0]

    if result == 1:
        st.write("Customer will churn ❌")
    else:
        st.write("Customer will not churn ✅")