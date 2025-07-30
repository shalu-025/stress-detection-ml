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
hr = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=70)
spo2 = st.number_input("SpO2 (%)", min_value=70, max_value=100, value=98)

# Predict button
if st.button("Detect Stress"):
    # Model expects 2D array: [[hr, spo2]]
    pred_encoded = model.predict([[hr, spo2]])[0]
    stress_level = label_encoder.inverse_transform([pred_encoded])[0]
    st.subheader(f"Stress Level: {stress_level}")
    suggestion = get_suggestion(stress_level)
    st.write(suggestion)
