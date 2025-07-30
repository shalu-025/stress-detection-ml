# stress-detection-ml
# Human Stress Detection using Machine Learning

This project predicts a user's stress level based on physiological inputs like heart rate and oxygen levels using a machine learning model (Random Forest).

## 📌 Features

- Takes input: Heart Rate, SpO2, etc.
- Predicts stress level: Low / Medium / High
- Gives personalized suggestions
- Simple Streamlit web interface
- Trained using Random Forest Classifier

## 📁 Folder Structure

HSD/
│
├── app/
│ ├── main.py # Streamlit frontend app
│ ├── suggestions.py # Suggestion logic
│
├── models/
│ ├── random_forest_model.pkl # Saved ML model
│ ├── label_encoder.pkl # LabelEncoder (if used)
│
├── train_model.ipynb # Jupyter notebook for model training
├── requirements.txt # List of dependencies
└── README.md # Project documentation
