# Customer Churn Prediction Streamlit App

This is a **Streamlit-based web application** that predicts customer churn using a pre-trained **Artificial Neural Network (ANN) model** saved in `.h5` format.

## Features
- User-friendly interface for entering customer details
- Pre-trained ANN model for churn prediction
- Real-time probability-based prediction
- Uses `StandardScaler` for feature normalization

## Installation
### 1. Clone the Repository
```bash
git clone https://github.com/Shishir880/Churn_Prediction.git
cd churn-prediction-app
```

### 3. Ensure Model and Scaler Files Exist
Make sure you have:
- `churn_prediction_model.h5` (Trained ANN model)
- `scaler.pkl` (Saved scaler object for normalization)

### 4. Run the App
```bash
streamlit run app.py
```

## Usage
1. Open the app in your browser.
2. Enter customer details (age, tenure, usage frequency, etc.).
3. Click on **Predict Churn**.
4. The app will display the probability of churn and provide a result.

## Dependencies
- `streamlit`
- `tensorflow`
- `numpy`
- `scikit-learn`
- `pickle`


## Author
Shishir Rahman

## License
This project is open-source under the MIT License.

