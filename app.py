import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# Feature names used during training
feature_names = [
    'scaled_time', 'scaled_amount',
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
    'V7', 'V8', 'V9', 'V10', 'V11', 'V12',
    'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
    'V19', 'V20', 'V21', 'V22', 'V23', 'V24',
    'V25', 'V26', 'V27', 'V28'
]

# Set up the Streamlit app
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.write("Enter the transaction details below to check if it's fraudulent.")

# Input form
user_input = []
with st.form("input_form"):
    for feature in feature_names:
        value = st.number_input(f"{feature}", value=0.0, format="%.6f")
        user_input.append(value)
    submitted = st.form_submit_button("Predict")

# Predict button logic
if submitted:
    try:
        # Prepare input
        input_df = pd.DataFrame([user_input], columns=feature_names)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1]

        # Show result
        if prediction == 1:
            st.error(f"⚠️ Fraudulent Transaction Detected! (Confidence: {proba:.2%})")
        else:
            st.success(f"✅ Legitimate Transaction. (Fraud Risk: {proba:.2%})")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
