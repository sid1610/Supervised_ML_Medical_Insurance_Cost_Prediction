# Medical Insurance Cost Prediction

This project predicts medical insurance cost from user details using a trained SVR model.

## Files

- `app.py`: Streamlit web app for prediction.
- `medical_insurance_model.pkl`: Trained machine learning model.
- `scaler.pkl`: Fitted scaler used before prediction.
- `requirements.txt`: Python packages needed to run the app.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

Use this repository and set the main file path to:

```text
app.py
```

## Inputs

The app uses age, sex, BMI, number of children, smoker status, and region. It also creates the engineered features expected by the saved model.

The saved model appears to predict log-transformed charges, so the app converts the prediction back to an estimated cost before displaying it.
