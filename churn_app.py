import streamlit as st
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

# ✅ Load the dataset from local file
@st.cache_data
def load_data():
    df = pd.read_csv("telco_churn.csv")  # Ensure this file is in the same folder
    df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    le = LabelEncoder()
    for col in df.select_dtypes(include='object'):
        df[col] = le.fit_transform(df[col])
    return df

# Load and preprocess
df = load_data()
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar for model selection
st.sidebar.title("🔧 Settings")
model_option = st.sidebar.selectbox("Choose Model", ["Logistic Regression", "Decision Tree"])
model = LogisticRegression(max_iter=1000) if model_option == "Logistic Regression" else DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# App title
st.title("📉 Customer Churn Prediction")
st.write(f"**Model Used:** `{model_option}` | **Accuracy:** `{accuracy_score(y_test, model.predict(X_test)):.2f}`")

# User input form
st.header("🧾 Enter Customer Details")

gender = st.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=2500.0)

# Prepare input data
input_data = pd.DataFrame({
    'gender': [0 if gender == "Female" else 1],
    'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
    'Partner': [1 if partner == "Yes" else 0],
    'Dependents': [1 if dependents == "Yes" else 0],
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
})

# Fill any missing columns with 0
for col in X.columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[X.columns]

# Prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    if prediction == 1:
        st.error(f"⚠️ High Risk of Churn! (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk of Churn (Probability: {probability:.2f})")
