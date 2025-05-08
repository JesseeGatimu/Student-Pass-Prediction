import streamlit as st
import numpy as np
import joblib

weights = joblib.load("weights.pkl")
biases = joblib.load("biases.pkl")
scaler = joblib.load("scaler.pkl")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_pass_fail(study_hours, previous_score):
    X = np.array([[study_hours, previous_score]])
    X_scaled = scaler.transform(X)
    z = np.dot(X_scaled, weights) + biases
    prob = sigmoid(z)
    return "Pass " if prob >= 0.5 else "Fail ", float(prob)

st.title("Student Exam Pass Predictor")
st.markdown("Enter your study hours and previous exam score:")

hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0, step=0.5)
previous_score = st.number_input("Previous Exam Score", min_value=0.0, max_value=100.0, step=1.0)

if st.button("Predict"):
    result, probability = predict_pass_fail(hours, previous_score)
    st.success(f"Prediction: **{result}**")
    st.info(f"Confidence: {probability * 100:.2f}%")
