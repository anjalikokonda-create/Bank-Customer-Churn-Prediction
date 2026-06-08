import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.title("🤖 Prediction")

df = pd.read_csv("data/raw/Churn_Modelling.csv")

# Encode categorical values
le_geo = LabelEncoder()
le_gender = LabelEncoder()

df["Geography"] = le_geo.fit_transform(df["Geography"])
df["Gender"] = le_gender.fit_transform(df["Gender"])

X = df.drop(["Exited", "RowNumber", "CustomerId", "Surname"], axis=1)
y = df["Exited"]

model = RandomForestClassifier()
model.fit(X, y)

st.subheader("Enter Customer Details")

credit_score = st.number_input("Credit Score", 300, 900)
age = st.number_input("Age", 18, 92)
balance = st.number_input("Balance", 0.0)
salary = st.number_input("Estimated Salary", 0.0)
products = st.slider("Number of Products", 1, 4)
active = st.selectbox("Is Active Member?", [0, 1])

data = [[credit_score, 0, 0, age, 5, balance, products, 1, active, salary]]

df_input = pd.DataFrame(data, columns=X.columns)

prediction = model.predict(df_input)

if st.button("Predict"):
    if prediction[0] == 1:
        st.error("Customer Will Churn ❌")
    else:
        st.success("Customer Will Stay ✔")
