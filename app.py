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


def inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --bg: #080d18;
            --panel: rgba(16, 29, 49, 0.82);
            --panel-strong: rgba(18, 37, 63, 0.95);
            --line: rgba(111, 231, 255, 0.24);
            --cyan: #38e8ff;
            --blue: #4b8dff;
            --text: #edf7ff;
            --muted: #9eb4c8;
        }

        .stApp {
            background:
                radial-gradient(circle at 20% 10%, rgba(56, 232, 255, 0.18), transparent 28%),
                radial-gradient(circle at 82% 18%, rgba(75, 141, 255, 0.16), transparent 30%),
                linear-gradient(135deg, #07111f 0%, #0a1020 45%, #111827 100%);
            color: var(--text);
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.4rem;
            padding-bottom: 2.4rem;
        }

        h1, h2, h3, p, label, span {
            color: var(--text);
        }

        .hero {
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 34px;
            min-height: 300px;
            background:
                linear-gradient(110deg, rgba(14, 25, 43, 0.92), rgba(15, 35, 59, 0.70)),
                repeating-linear-gradient(135deg, rgba(56, 232, 255, 0.08) 0 1px, transparent 1px 42px);
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.38), inset 0 0 38px rgba(56, 232, 255, 0.06);
            position: relative;
            overflow: hidden;
        }

        .hero:after {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, transparent 0 48%, rgba(56, 232, 255, 0.09) 49% 51%, transparent 52%),
                linear-gradient(0deg, transparent 0 48%, rgba(75, 141, 255, 0.08) 49% 51%, transparent 52%);
            background-size: 150px 150px;
            opacity: 0.32;
            pointer-events: none;
        }

        .eyebrow {
            color: var(--cyan);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.18rem;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .hero-title {
            color: var(--text);
            font-size: 3.2rem;
            font-weight: 800;
            line-height: 1.05;
            margin: 0;
            text-shadow: 0 0 22px rgba(56, 232, 255, 0.35);
        }

        .hero-copy {
            color: var(--muted);
            max-width: 620px;
            font-size: 1.04rem;
            margin-top: 18px;
        }

        .status-row {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 26px;
        }

        .status-pill {
            border: 1px solid rgba(56, 232, 255, 0.32);
            border-radius: 999px;
            padding: 9px 14px;
            background: rgba(56, 232, 255, 0.08);
            color: #cdf8ff;
            font-size: 0.88rem;
        }

        .glass-panel {
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 24px;
            background: var(--panel);
            box-shadow: 0 18px 60px rgba(0, 0, 0, 0.28);
        }

        .panel-title {
            color: var(--cyan);
            font-size: 0.86rem;
            font-weight: 800;
            letter-spacing: 0.14rem;
            text-transform: uppercase;
            margin-bottom: 14px;
        }

        .metric-card {
            border: 1px solid rgba(56, 232, 255, 0.30);
            border-radius: 16px;
            padding: 24px;
            background: linear-gradient(145deg, rgba(56, 232, 255, 0.12), rgba(75, 141, 255, 0.08));
            min-height: 186px;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.86rem;
            text-transform: uppercase;
            letter-spacing: 0.1rem;
        }

        .metric-value {
            color: #ffffff;
            font-size: 2.65rem;
            font-weight: 850;
            margin: 8px 0 6px;
            text-shadow: 0 0 28px rgba(56, 232, 255, 0.55);
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.92rem;
        }

        .system-card {
            border: 1px solid rgba(56, 232, 255, 0.22);
            border-radius: 16px;
            padding: 22px;
            min-height: 222px;
            background:
                radial-gradient(circle at 70% 30%, rgba(56, 232, 255, 0.24), transparent 16%),
                linear-gradient(145deg, rgba(17, 34, 57, 0.92), rgba(13, 24, 39, 0.80));
            position: relative;
            overflow: hidden;
        }

        .network-line {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(56, 232, 255, 0.7), transparent);
            margin: 18px 0;
        }

        .node-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            margin-top: 20px;
        }

        .node {
            border: 1px solid rgba(56, 232, 255, 0.24);
            border-radius: 12px;
            min-height: 72px;
            padding: 12px;
            background: rgba(255, 255, 255, 0.035);
        }

        .node strong {
            color: #dbfbff;
            display: block;
            font-size: 0.95rem;
        }

        .node span {
            color: var(--muted);
            font-size: 0.78rem;
        }

        .stButton > button {
            width: 100%;
            border: 1px solid rgba(56, 232, 255, 0.55);
            border-radius: 12px;
            background: linear-gradient(90deg, #06b6d4, #2563eb);
            color: white;
            font-weight: 800;
            padding: 0.82rem 1rem;
            box-shadow: 0 12px 34px rgba(37, 99, 235, 0.35);
        }

        .stButton > button:hover {
            border-color: #aaf7ff;
            box-shadow: 0 14px 42px rgba(56, 232, 255, 0.36);
        }

        div[data-testid="stForm"] {
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 22px;
            background: var(--panel-strong);
            box-shadow: 0 18px 55px rgba(0, 0, 0, 0.22);
        }

        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input {
            background-color: rgba(255, 255, 255, 0.06);
            border-color: rgba(56, 232, 255, 0.22);
            color: white;
        }

        .stSlider [data-baseweb="slider"] {
            padding-top: 0.8rem;
        }

        @media (max-width: 760px) {
            .hero {
                padding: 24px;
                min-height: auto;
            }

            .hero-title {
                font-size: 2.25rem;
            }

            .node-row {
                grid-template-columns: 1fr 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Medical Insurance Cost Prediction",
        layout="wide",
    )
    inject_styles()

    st.markdown(
        """
        <section class="hero">
            <div class="eyebrow">Streamlit ML Deployment</div>
            <h1 class="hero-title">Medical Insurance<br>Cost Predictor</h1>
            <p class="hero-copy">
                Enter patient profile details and run the trained SVR model to estimate
                medical insurance charges in real time.
            </p>
            <div class="status-row">
                <div class="status-pill">Model: SVR</div>
                <div class="status-pill">Scaler: StandardScaler</div>
                <div class="status-pill">Status: Ready</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    form_col, output_col = st.columns([0.95, 1.05], gap="large")

    with form_col:
        st.markdown('<div class="panel-title">Prediction Inputs</div>', unsafe_allow_html=True)
        with st.form("prediction_form"):
            age = st.slider("Age", min_value=18, max_value=100, value=30)
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

            left, right = st.columns(2)
            with left:
                sex = st.selectbox("Sex", ["Female", "Male"])
                children = st.slider("Children", min_value=0, max_value=10, value=0)
            with right:
                smoker = st.selectbox("Smoker", ["No", "Yes"])
                region = st.selectbox(
                    "Region",
                    ["Northeast", "Northwest", "Southeast", "Southwest"],
                )

            submitted = st.form_submit_button("Run Prediction")

    prediction = None
    if submitted:
        try:
            model, scaler = load_artifacts()
            features = build_feature_frame(age, sex, bmi, children, smoker, region)
            scaled_features = pd.DataFrame(
                scaler.transform(features),
                columns=FEATURE_COLUMNS,
            )
            log_prediction = model.predict(scaled_features)[0]
            prediction = math.expm1(log_prediction)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    with output_col:
        st.markdown('<div class="panel-title">System Overview</div>', unsafe_allow_html=True)
        if prediction is None:
            value = "Awaiting Run"
            note = "Fill in the profile and click Run Prediction."
        else:
            value = f"${prediction:,.2f}"
            note = "Estimated annual insurance cost from the trained model."

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Charge</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown(
            f"""
            <div class="system-card">
                <div class="panel-title">Feature Signal</div>
                <div class="network-line"></div>
                <div class="node-row">
                    <div class="node"><strong>{age}</strong><span>Age</span></div>
                    <div class="node"><strong>{bmi:.1f}</strong><span>BMI</span></div>
                    <div class="node"><strong>{smoker}</strong><span>Smoker</span></div>
                    <div class="node"><strong>{region}</strong><span>Region</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
