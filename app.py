import streamlit as st

from src.utils import load_artifact, preprocess_data
from src.pipelines.predict_pipeline import CustomData, PredictPipeline

parameters = load_artifact(r"./conf/parameters.yml")
data = preprocess_data(r"./artifacts/raw_data.parquet")


st.title("Income Evaluation")

if st.checkbox("Show Dataset"):
    st.dataframe(data)

st.write("### Required Information:")

# categorical features
education_levels = parameters["ordinal"]["categories"]["education"]
education = st.selectbox("Education", education_levels)

occupations = sorted(set(data["occupation"].dropna()))
occupation = st.selectbox("Occupation", occupations)

relationships = [
    "Husband",
    "Wife",
    "Own-child",
    "Other-relative",
    "Unmarried",
    "Not-in-family"
]
relationship = st.selectbox("Relationship", relationships)

marital_statuses = [
    "Married-AF-spouse",
    "Married-civ-spouse",
    "Married-spouse-absent",
    "Never-married",
    "Separated",
    "Divorced",
    "Widowed"
]
marital_status = st.selectbox("Marital Status", marital_statuses)

capital_gains = ["Yes", "No"]
capital_gain = st.selectbox("Capital Gains", capital_gains)

# numeric features
age = st.slider(
    "Age",
    int(data["age"].min()),
    int(data["age"].max()),
    20
)

button = st.button("Predict Income Class")

if button:
    record = CustomData(
        education=education,
        occupation=occupation,
        relationship=relationship,
        marital_status=marital_status,
        capital_gain=capital_gain,
        age=age
    ).to_dataframe()
    prediction = PredictPipeline().predict(record)
    mapper = {
        "<=50K": "This individual does not make more than $50,000 annually",
        ">50K": "This individual makes more than $50,000 annually"
    }
    st.subheader(prediction.replace(prediction, mapper[prediction]))
