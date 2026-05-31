import math
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "medical_insurance_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

FEATURE_COLUMNS = [
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region_northwest",
    "region_southeast",
    "region_southwest",
    "bmi_risk",
    "age_bmi_interaction",
]


@st.cache_resource
def load_artifacts():
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with SCALER_PATH.open("rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler


def build_feature_frame(age, sex, bmi, children, smoker, region):
    sex_value = 1 if sex == "Male" else 0
    smoker_value = 1 if smoker == "Yes" else 0

    data = {
        "age": age,
        "sex": sex_value,
        "bmi": bmi,
        "children": children,
        "smoker": smoker_value,
        "region_northwest": 1 if region == "Northwest" else 0,
        "region_southeast": 1 if region == "Southeast" else 0,
        "region_southwest": 1 if region == "Southwest" else 0,
        "bmi_risk": 1 if bmi >= 30 else 0,
        "age_bmi_interaction": age * bmi,
    }
    return pd.DataFrame([data], columns=FEATURE_COLUMNS)


def main():
    st.set_page_config(
        page_title="Medical Insurance Cost Prediction",
        layout="centered",
    )

    st.title("Medical Insurance Cost Prediction")

    with st.form("prediction_form"):
        age = st.slider("Age", min_value=18, max_value=100, value=30)
        sex = st.selectbox("Sex", ["Female", "Male"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        children = st.slider("Number of children", min_value=0, max_value=10, value=0)
        smoker = st.selectbox("Smoker", ["No", "Yes"])
        region = st.selectbox(
            "Region",
            ["Northeast", "Northwest", "Southeast", "Southwest"],
        )

        submitted = st.form_submit_button("Predict cost")

    if submitted:
        model, scaler = load_artifacts()
        features = build_feature_frame(age, sex, bmi, children, smoker, region)
        scaled_features = pd.DataFrame(
            scaler.transform(features),
            columns=FEATURE_COLUMNS,
        )
        log_prediction = model.predict(scaled_features)[0]
        prediction = math.expm1(log_prediction)

        st.subheader("Estimated Insurance Cost")
        st.success(f"${prediction:,.2f}")


if __name__ == "__main__":
    main()
