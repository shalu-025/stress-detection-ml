import streamlit as st
import joblib
from app.suggestions import get_suggestion
import os

# Load model and encoder
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "random_forest_model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "label_encoder.pkl")
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

st.title("Human Stress Detection")

# Input fields
# User Inputs
age = st.number_input("Enter your age:", min_value=1, max_value=120, step=1)
gender = st.selectbox("Select Gender:", ["Male", "Female"])
hr = st.number_input("Enter Heart Rate (HR):", min_value=30, max_value=200)
spo2 = st.number_input("Enter SpO2 (%):", min_value=70, max_value=100)

# Default average values (you can customize these)
temp_avg = 98.6
bp_sys_avg = 120
bp_dia_avg = 80
resp_rate_avg = 16

# Prepare feature vector
features = [[age, 0 if gender == "Male" else 1, hr, spo2, temp_avg, bp_sys_avg, bp_dia_avg, resp_rate_avg]]


# Predict button
if st.button("Detect Stress"):
    try:
        # Full feature vector: [age, gender_encoded, hr, spo2, temp, sys_bp, dia_bp, resp_rate]
        input_data = [[age, 0 if gender == "Male" else 1, hr, spo2, temp_avg, bp_sys_avg, bp_dia_avg, resp_rate_avg]]
        
        # Make prediction
        pred_encoded = model.predict(input_data)[0]
        stress_level = label_encoder.inverse_transform([pred_encoded])[0]

        # Display result
        st.subheader(f"Stress Level: {stress_level}")
        suggestion = get_suggestion(stress_level)
        st.write(suggestion)

    except Exception as e:
        st.error(f"Something went wrong: {e}")

