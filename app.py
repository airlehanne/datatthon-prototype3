import streamlit as st
import pandas as pd
import pickle

# ---------------------------------------------------
# Load model files
# ---------------------------------------------------

with open("brfss_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("brfss_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("brfss_features.pkl", "rb") as f:
    FEATURES = pickle.load(f)

# ---------------------------------------------------
# App Title
# ---------------------------------------------------

st.title("Diabetes Risk Assessment Tool")

st.write("""
This AI prototype estimates diabetes risk using
self-reported lifestyle and health information.
""")

# ---------------------------------------------------
# User Inputs
# ---------------------------------------------------

st.header("Patient Information")

HighBP = st.selectbox("High Blood Pressure", [0,1])
HighChol = st.selectbox("High Cholesterol", [0,1])
CholCheck = st.selectbox("Cholesterol Check", [0,1])

BMI = st.slider("BMI", 10.0, 70.0, 25.0)

Smoker = st.selectbox("Smoking History", [0,1])
Stroke = st.selectbox("History of Stroke", [0,1])

HeartDiseaseorAttack = st.selectbox(
    "Heart Disease / Heart Attack",
    [0,1]
)

PhysActivity = st.selectbox("Physical Activity", [0,1])
NoDocbcCost = st.selectbox(
    "Could Not See Doctor Due To Cost",
    [0,1]
)

GenHlth = st.slider(
    "General Health (1 = Excellent, 5 = Poor)",
    1, 5, 3
)

MentHlth = st.slider(
    "Poor Mental Health Days",
    0, 30, 0
)

PhysHlth = st.slider(
    "Poor Physical Health Days",
    0, 30, 0
)

DiffWalk = st.selectbox(
    "Difficulty Walking",
    [0,1]
)
Sex = st.selectbox(
    "Sex (0 = Female, 1 = Male)",
    [0,1]
)

Age = st.slider(
    "Age Group (1–13)",
    1, 13, 5
)

Education = st.slider(
    "Education Level",
    1, 6, 4
)

Income = st.slider(
    "Income Level",
    1, 8, 4
)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

if st.button("Predict Diabetes Risk"):

    input_data = pd.DataFrame([[
        HighBP,
        HighChol,
        CholCheck,
        BMI,
        Smoker,
        Stroke,
        HeartDiseaseorAttack,
        PhysActivity,
        NoDocbcCost,
        GenHlth,
        MentHlth,
        PhysHlth,
        DiffWalk,
        Sex,
        Age,
        Education,
        Income
    ]], columns=FEATURES)


# Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0][1]

    # Output
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"Higher Diabetes Risk ({probability*100:.1f}%)"
        )

    else:
        st.success(
            f"Lower Diabetes Risk ({probability*100:.1f}%)"
        )

    st.write(
        "This tool is for educational purposes only "
        "and does not replace medical advice."
    )