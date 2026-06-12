"""
Phone Addiction Level Predictor
================================
Loads the saved GradientBoosting sklearn Pipeline (model.pkl) and exposes
a clean prediction interface via a Streamlit web app.

Run:
    pip install -r requirements.txt streamlit
    streamlit run app.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

CATEGORICAL_COLS = ["gender", "phone_usage_purpose"]

NUMERICAL_COLS = [
    "age",
    "daily_usage_hours",
    "sleep_hours",
    "interllectual_performance",
    "social_interactions",
    "exercise_hours",
    "anxiety_level",
    "depression_level",
    "self_esteem",
    "screen_time_before_bed",
    "phone_checks_per_day",
    "apps_used_daily",
    "time_on_social_media",
    "time_on_gaming",
    "time_on_education",
    "family_communication",
    "weekend_usage_hours",
]

ALL_FEATURES = CATEGORICAL_COLS + NUMERICAL_COLS

GENDER_OPTIONS         = ["Male", "Female", "Other"]
USAGE_PURPOSE_OPTIONS  = ["Social Media", "Education", "Entertainment", "Work", "Gaming", "Communication"]

ADDICTION_LABELS = {
    (0.0, 2.0): ("🟢 Low",      "You show minimal signs of phone addiction."),
    (2.0, 3.5): ("🟡 Moderate", "You show moderate phone usage patterns. Consider digital wellness practices."),
    (3.5, 5.0): ("🔴 High",     "You show strong signs of phone addiction. Consider seeking support."),
}


# ─────────────────────────────────────────────
# PIPELINE LOADER
# ─────────────────────────────────────────────
@st.cache_resource
def load_pipeline() -> object:
    """Load the trained sklearn Pipeline from disk (cached across sessions)."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()
    pipeline = joblib.load(MODEL_PATH)
    return pipeline


# ─────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────
def predict_addiction(pipeline, input_data: dict) -> float:
    """
    Run inference through the full sklearn Pipeline.

    Parameters
    ----------
    pipeline   : fitted sklearn Pipeline (preprocessor → model)
    input_data : dict with keys matching ALL_FEATURES

    Returns
    -------
    float : predicted addiction_level
    """
    df_input = pd.DataFrame([input_data], columns=ALL_FEATURES)
    prediction = pipeline.predict(df_input)[0]
    return float(np.clip(prediction, 0.0, 5.0))


def get_addiction_label(score: float) -> tuple[str, str]:
    for (low, high), (label, msg) in ADDICTION_LABELS.items():
        if low <= score < high:
            return label, msg
    return ("🔴 High", "You show strong signs of phone addiction.")


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Phone Addiction Predictor",
        page_icon="📱",
        layout="centered",
    )

    st.title("📱 Phone Addiction Level Predictor")
    st.markdown(
        "Fill in the details below and click **Predict** to estimate your "
        "phone addiction level (0 = none, 5 = severe)."
    )

    pipeline = load_pipeline()

    # ── Sidebar: model info ──────────────────────────────────────────────
    with st.sidebar:
        st.header("ℹ️ Model Info")
        st.write("**Algorithm:** Gradient Boosting Regressor")
        st.write("**Preprocessor:** OneHotEncoder + StandardScaler")
        st.write("**Target:** Addiction Level (0 – 5)")
        st.markdown("---")
        st.caption("Model trained with Optuna HPT via MLflow.")

    # ── Input form ───────────────────────────────────────────────────────
    with st.form("prediction_form"):
        st.subheader("👤 Demographics")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", GENDER_OPTIONS)
        with col2:
            age = st.number_input("Age", min_value=10, max_value=80, value=22, step=1)

        st.subheader("📱 Phone Usage")
        col3, col4 = st.columns(2)
        with col3:
            phone_usage_purpose = st.selectbox("Primary Usage Purpose", USAGE_PURPOSE_OPTIONS)
            daily_usage_hours   = st.slider("Daily Usage Hours", 0.0, 16.0, 4.0, 0.5)
            weekend_usage_hours = st.slider("Weekend Usage Hours", 0.0, 18.0, 5.0, 0.5)
            screen_time_before_bed = st.slider("Screen Time Before Bed (hrs)", 0.0, 5.0, 1.0, 0.25)
        with col4:
            phone_checks_per_day  = st.number_input("Phone Checks / Day", 0, 300, 40)
            apps_used_daily       = st.number_input("Apps Used Daily", 1, 50, 8)
            time_on_social_media  = st.slider("Time on Social Media (hrs)", 0.0, 10.0, 2.0, 0.25)
            time_on_gaming        = st.slider("Time on Gaming (hrs)", 0.0, 8.0, 0.5, 0.25)
            time_on_education     = st.slider("Time on Education (hrs)", 0.0, 8.0, 0.5, 0.25)

        st.subheader("🧠 Health & Lifestyle")
        col5, col6 = st.columns(2)
        with col5:
            sleep_hours             = st.slider("Sleep Hours / Night", 2.0, 12.0, 7.0, 0.5)
            exercise_hours          = st.slider("Exercise Hours / Day", 0.0, 5.0, 0.5, 0.25)
            family_communication    = st.slider("Family Communication (hrs/day)", 0.0, 6.0, 1.0, 0.25)
            social_interactions     = st.number_input("Social Interactions / Day", 0, 50, 5)
        with col6:
            anxiety_level           = st.slider("Anxiety Level (0–10)", 0, 10, 4)
            depression_level        = st.slider("Depression Level (0–10)", 0, 10, 3)
            self_esteem             = st.slider("Self-Esteem (0–10)", 0, 10, 6)
            interllectual_performance = st.slider("Intellectual Performance (0–10)", 0, 10, 6)

        submitted = st.form_submit_button("🔍 Predict Addiction Level", use_container_width=True)

    # ── Prediction output ────────────────────────────────────────────────
    if submitted:
        input_data = {
            "gender":                   gender,
            "phone_usage_purpose":      phone_usage_purpose,
            "age":                      age,
            "daily_usage_hours":        daily_usage_hours,
            "sleep_hours":              sleep_hours,
            "interllectual_performance": interllectual_performance,
            "social_interactions":      social_interactions,
            "exercise_hours":           exercise_hours,
            "anxiety_level":            anxiety_level,
            "depression_level":         depression_level,
            "self_esteem":              self_esteem,
            "screen_time_before_bed":   screen_time_before_bed,
            "phone_checks_per_day":     phone_checks_per_day,
            "apps_used_daily":          apps_used_daily,
            "time_on_social_media":     time_on_social_media,
            "time_on_gaming":           time_on_gaming,
            "time_on_education":        time_on_education,
            "family_communication":     family_communication,
            "weekend_usage_hours":      weekend_usage_hours,
        }

        score = predict_addiction(pipeline, input_data)
        label, message = get_addiction_label(score)

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        col_score, col_label = st.columns([1, 2])
        with col_score:
            st.metric("Addiction Score", f"{score:.2f} / 5.00")
        with col_label:
            st.markdown(f"**Level:** {label}")
            st.caption(message)

        st.progress(score / 5.0)

        with st.expander("🔎 View Input Summary"):
            st.dataframe(
                pd.DataFrame([input_data]).T.rename(columns={0: "Value"}),
                use_container_width=True,
            )


if __name__ == "__main__":
    main()