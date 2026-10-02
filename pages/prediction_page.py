import streamlit as st
import pandas as pd
import joblib


# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Ad Click Prediction",
    page_icon="🔮",
    layout="wide"
)


# Load dataset
df = pd.read_csv("cleaned.csv")

countries = sorted(df["Country"].dropna().unique())


# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    return joblib.load("final_model.pkl")


model = load_model()


# =========================
# Page Header
# =========================
st.title("🔮 Ad Click Prediction")

st.markdown("""
Enter the user's information below to predict whether
the user is likely to click on the advertisement.
""")

st.divider()


# =========================
# User Inputs
# =========================
st.header("👤 User Information")

col1, col2 = st.columns(2)

with col1:

    daily_time = st.number_input(
        "Daily Time Spent on Site (minutes)",
        min_value=0.0,
        max_value=200.0,
        value=60.0,
        step=0.01
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=30,
        step=1
    )

    area_income = st.number_input(
        "Area Income",
        min_value=0.0,
        value=50000.0,
        step=100.0
    )


with col2:

    daily_internet = st.number_input(
        "Daily Internet Usage (minutes)",
        min_value=0.0,
        max_value=1000.0,
        value=200.0,
        step=0.01
    )

    male = st.selectbox(
        "Gender",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
     )

    country = st.selectbox(
        "Country",
        options=countries
    )


st.divider()


# =========================
# Prediction
# =========================
if st.button("🔮 Predict", use_container_width=True):

    # Create DataFrame
    input_data = pd.DataFrame({
        "Daily Time Spent on Site": [daily_time],
        "Age": [age],
        "Area Income": [area_income],
        "Daily Internet Usage": [daily_internet],
        "Male": [male],
        "Country": [country]
    })

    try:

        # Prediction
        prediction = model.predict(input_data)[0]

        # Probability
        probability = model.predict_proba(input_data)[0]

        st.divider()

        st.header("📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:
                st.error("🔴 User is likely to CLICK the advertisement")
            else:
                st.success("🟢 User is likely NOT to click the advertisement")

        with col2:

            st.metric(
                "Probability of Clicking",
                f"{probability[1] * 100:.2f}%"
            )

        # Show probabilities
        st.subheader("Prediction Probabilities")

        result_df = pd.DataFrame({
            "Class": ["No Click (0)", "Click (1)"],
            "Probability": [
                f"{probability[0] * 100:.2f}%",
                f"{probability[1] * 100:.2f}%"
            ]
        })

        st.table(result_df)

    except Exception as e:

        st.error("Prediction failed.")

        st.exception(e)