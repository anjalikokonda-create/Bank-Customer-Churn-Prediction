import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data/raw/Churn_Modelling.csv")

df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

le_geo = LabelEncoder()
df["Geography"] = le_geo.fit_transform(df["Geography"])

le_gender = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])

X = df.drop("Exited", axis=1)
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "models/final_model.joblib")
joblib.dump(scaler, "models/preprocessor.joblib")

print("Model trained and saved successfully!")