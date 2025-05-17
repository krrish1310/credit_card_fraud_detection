import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💳 Credit Card Fraud Detection")
st.write("Upload a transaction CSV or enter values manually to check for fraud.")

uploaded_file = st.file_uploader("Upload a CSV with same structure", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    st.write("Prediction for first row:", "Fraud ❌" if prediction[0] == 1 else "Legit ✅")

st.subheader("Manual Input")
v_features = []
for i in range(1, 29):
    val = st.number_input(f"V{i}", value=0.0)
    v_features.append(val)
scaled_amount = st.number_input("Scaled Amount", value=0.0)
scaled_time = st.number_input("Scaled Time", value=0.0)

if st.button("Predict"):
    input_data = pd.DataFrame([[scaled_time, scaled_amount] + v_features])
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    st.success("Prediction: " + ("Fraud ❌" if pred == 1 else "Legit ✅"))
