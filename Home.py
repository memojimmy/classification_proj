
import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt


# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Ad Click Prediction",
    page_icon="📊",
    layout="wide"
)

# =========================
# Header
# =========================
st.title("📊 Ad Click Prediction")
st.subheader("Machine Learning Project")

st.markdown("""
Welcome to the **Ad Click Prediction** project.

This application uses Machine Learning to predict whether a user
will **click on an advertisement** based on their demographic
information and internet usage behavior.
""")

st.divider()

# =========================
# Project Overview
# =========================
st.header("📌 Project Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset Size", "1000 Rows")

with col2:
    st.metric("Features", "7")

with col3:
    st.metric("Target", "Clicked on Ad")

st.markdown("""
### 🎯 Project Goal

The main goal of this project is to predict whether a user will
click on an advertisement using features such as:

- Daily time spent on the website
- Age
- Area income
- Daily internet usage
- Gender
- Country
""")

st.divider()

# =========================
# Features Description
# =========================
st.header("📋 Features Description")

data = {
    "Column": [
        "Unnamed: 0",
        "Daily Time Spent on Site",
        "Age",
        "Area Income",
        "Daily Internet Usage",
        "Male",
        "Country",
        "Clicked on Ad"
    ],

    "Description": [
        "Unique index for each record.",
        "Average daily time spent by the user on the website, measured in minutes.",
        "Age of the user in years.",
        "Average income of the user's geographical area.",
        "Average daily internet usage of the user, measured in minutes.",
        "Gender indicator: 0 = Female, 1 = Male.",
        "Country where the user is located.",
        "Target variable: 0 = Did not click the advertisement, 1 = Clicked the advertisement."
    ],

    "Type": [
        "Integer",
        "Float",
        "Integer",
        "Float",
        "Float",
        "Integer",
        "Categorical",
        "Binary"
    ],

    "Role": [
        "Index",
        "Feature",
        "Feature",
        "Feature",
        "Feature",
        "Feature",
        "Feature",
        "Target"
    ]
}

st.dataframe(
    data,
    use_container_width=True,
    hide_index=True
)

st.divider()

# =========================
# Target Variable
# =========================
st.header("🎯 Target Variable")

st.info("""
**Clicked on Ad**

- **0** → The user did not click on the advertisement.
- **1** → The user clicked on the advertisement.
""")

st.divider()

# =========================
# Machine Learning Workflow
# =========================
st.header("⚙️ Machine Learning Workflow")

st.markdown("""
The project follows these main steps:

**1️⃣ Data Collection**  
↓  
**2️⃣ Data Cleaning & Preprocessing**  
↓  
**3️⃣ Exploratory Data Analysis (EDA)**  
↓  
**4️⃣ Feature Engineering**  
↓  
**5️⃣ Model Training**  
↓  
**6️⃣ Model Evaluation**  
↓  
**7️⃣ Prediction**
""")

st.divider()

# =========================
# Navigation
# =========================
st.header("🚀 Start Prediction")

st.write(
    "Use the prediction page to enter user information "
    "and get the model prediction."
)

if st.button("🔮 Go to Prediction"):
    st.switch_page("classification_proj/prediction_page.py")

# =========================
# Footer
# =========================
st.markdown("---")

st.caption("Machine Learning Project | Ad Click Prediction")
