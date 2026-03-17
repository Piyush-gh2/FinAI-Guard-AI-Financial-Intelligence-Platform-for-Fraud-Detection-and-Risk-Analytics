import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

# Load data
data = pd.read_csv("data/transactions.csv")

# Encode categorical columns
le = LabelEncoder()
data["Customer_ID"] = le.fit_transform(data["Customer_ID"])
data["Location"] = le.fit_transform(data["Location"])
data["Device"] = le.fit_transform(data["Device"])

# Features & target
X = data.drop(["Fraud_Label","Transaction_ID"], axis=1)
y = data["Fraud_Label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = XGBClassifier(
    n_estimators=120,
    learning_rate=0.1,
    max_depth=5
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model,"models/fraud_model.pkl")

print("Model trained successfully")