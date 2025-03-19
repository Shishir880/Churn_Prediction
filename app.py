import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle

# Load trained model
model = load_model("churn_prediction_model.h5")

# Load scaler (if used during training)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("Customer Churn Prediction App")
st.write("Enter customer details to predict churn.")

# User input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=24)
usage_freq = st.number_input("Usage Frequency", min_value=0, max_value=50, value=10)
support_calls = st.number_input("Support Calls", min_value=0, max_value=20, value=5)
payment_delay = st.number_input("Payment Delay (days)", min_value=0, max_value=30, value=10)
total_spend = st.number_input("Total Spend ($)", min_value=0.0, max_value=10000.0, value=500.0)
last_interaction = st.number_input("Last Interaction (days)", min_value=0, max_value=365, value=30)

gender = st.selectbox("Gender", ["Male", "Female"])
subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])

# Encoding categorical variables
gender_dict = {"Male": 1, "Female": 0}
subscription_dict = {"Basic": 0, "Standard": 1, "Premium": 2}
contract_dict = {"Monthly": 0, "Quarterly": 1, "Annual": 2}

gender = gender_dict[gender]
subscription_type = subscription_dict[subscription_type]
contract_length = contract_dict[contract_length]

# Prepare input data
input_data = np.array([[age, tenure, usage_freq, support_calls, payment_delay, total_spend,
                        last_interaction, gender, subscription_type, contract_length]])

# Scale input data
input_data = scaler.transform(input_data)

# Predict churn
if st.button("Predict Churn"):
    prediction = model.predict(input_data)
    churn_prob = prediction[0][0]
    if churn_prob > 0.5:
        st.error(f"High chance of churn! (Probability: {churn_prob:.2f})")
    else:
        st.success(f"Low chance of churn. (Probability: {churn_prob:.2f})")
