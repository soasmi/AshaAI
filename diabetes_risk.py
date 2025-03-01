import pandas as pd
import numpy as np
import hashlib
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset with basic health indicators
data = {
    'Age': [45, 50, 35, 60, 40, 55],
    'BMI': [28.5, 30.2, 25.0, 31.5, 26.8, 29.0],
    'BloodPressure': [120, 135, 110, 145, 125, 138],
    'Glucose': [140, 160, 110, 180, 130, 155],
    'FamilyHistory': [1, 1, 0, 1, 0, 1],  # 1 = Yes, 0 = No
    'PhysicalActivity': [3, 1, 5, 0, 4, 2],  # Days per week
    'DiabetesRisk': [1, 1, 0, 1, 0, 1]  # 1 = High risk, 0 = Low risk
}

df = pd.DataFrame(data)

# Splitting data
X = df.drop(columns=['DiabetesRisk'])
y = df['DiabetesRisk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Explainability with SHAP
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Plot SHAP values for feature importance
shap.summary_plot(shap_values, X_test)

# Secure Data Logging using SHA-256
def secure_log(patient_data):
    patient_str = str(patient_data)
    hashed_data = hashlib.sha256(patient_str.encode()).hexdigest()
    return hashed_data

# Example patient record
new_patient = {'Age': 42, 'BMI': 27.5, 'BloodPressure': 125, 'Glucose': 135, 'FamilyHistory': 1, 'PhysicalActivity': 2}
print(f"Secure Patient Data Hash: {secure_log(new_patient)}")

# Save model
joblib.dump(model, 'diabetes_risk_model.pkl')
