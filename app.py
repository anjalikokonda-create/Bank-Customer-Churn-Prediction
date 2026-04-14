st.write("Restarting app...")
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bank Churn Prediction")

# Load model and scaler
model = joblib.load("models/final_model.joblib")
scaler = joblib.load("models/preprocessor.joblib")

st.title("🏦 Bank Customer Churn Prediction")

# ---------------- INPUTS ----------------
credit_score = st.number_input("Credit Score", 300, 900)
age = st.number_input("Age", 18, 100)
balance = st.number_input("Balance")
estimated_salary = st.number_input("Estimated Salary")

# ---------------- PREDICT ----------------
if st.button("Predict"):

    # FIXED FEATURE ORDER (MATCH TRAINING DATA)
    data = pd.DataFrame([[credit_score, 0, 1, age, 5, balance, 1, 1, 1, estimated_salary]],
                        columns=[
                            "CreditScore", "Geography", "Gender", "Age",
                            "Tenure", "Balance", "NumOfProducts",
                            "HasCrCard", "IsActiveMember", "EstimatedSalary"
                        ])

    # Scale data
    data = scaler.transform(data)

    # Prediction
    result = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    # ---------------- OUTPUT ----------------
    if result == 1:
        st.error("Customer will churn ❌")
    else:
        st.success("Customer will NOT churn ✅")

    st.write(f"Churn Probability: {probability * 100:.2f}%")

    # ---------------- RISK LEVEL ----------------
    if probability < 0.40:
        st.success("Risk Level: 🟢 LOW RISK")
    elif probability < 0.70:
        st.warning("Risk Level: 🟠 MEDIUM RISK")
    else:
        st.error("Risk Level: 🔴 HIGH RISK")

    # ---------------- CHART 1 ----------------
    st.subheader("📊 Input Chart")

    features = ["Credit Score", "Age", "Balance", "Estimated Salary"]
    values = [credit_score, age, balance, estimated_salary]

    fig, ax = plt.subplots()
    ax.bar(features, values)
    st.pyplot(fig)

    # ---------------- CHART 2 ----------------
    st.subheader("📈 Churn Probability Chart")

    fig2, ax2 = plt.subplots()
    ax2.bar(["Not Churn", "Churn"], [1 - probability, probability])
    st.pyplot(fig2)