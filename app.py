import streamlit as st
import plotly.express as px

from streamlit_pandas_profiling import st_profile_report

from src.utils import (
    load_artifact,
    preprocess_data,
    create_profile_report
)
from src.pipelines.predict_pipeline import CustomData, PredictPipeline


st.set_page_config(layout="wide")
data = preprocess_data(r"./artifacts/raw_data.parquet")
parameters = load_artifact(r"./conf/parameters.yml")
target = parameters["target"]


def create_app():
    """
    Creates a Steamlit web application
    """
    st.title("Income Classification :moneybag:")
    selection = st.sidebar.radio(
        "Make a selection :point_down:",
        options=["Analyze", "Visualize", "Predict"]
    )

    if selection == "Analyze":
        if st.checkbox("Show Dataset"):
            st.dataframe(data)
        if st.checkbox("Generate Report"):
            profile = create_profile_report(data)
            st_profile_report(profile)

    if selection == "Visualize":
        if st.checkbox("How are gender and race related to income class?"):
            df_eda = (
                data
                .groupby(["gender", "race", target])
                .size()
                .reset_index(name="count")
            )

            # normalize the 'count' feature
            df_eda["normalized_count"] = df_eda["count"] / data.shape[0]
            fig = px.treemap(
                df_eda,
                path=df_eda.drop("count", axis=1).columns[:-1],
                values=df_eda.columns[-1],
                color=target
            )
            fig.update_layout(
                margin=dict(t=0, r=0, b=0, l=0),
                height=1000,
                width=1300
            )
            st.plotly_chart(fig)
        if st.checkbox("How is income class related to gender and education?"):
            df_eda = (
                data
                .groupby([target, "gender", "education"])
                .size()
                .reset_index(name="count")
            )

            # normalize the 'count' feature
            df_eda["normalized_count"] = df_eda["count"] / data.shape[0]

            fig = px.sunburst(
                df_eda,
                path=df_eda.drop("count", axis=1).columns[:-1],
                values=df_eda.columns[-1]
            )
            fig.update_layout(
                height=1500,
                width=1500
            )
            st.plotly_chart(fig)
        if st.checkbox("How are capital gains related to income class and occupation?"):
            df_eda = (
                data
                .groupby(["capital_gain", target, "occupation"])
                .size()
                .reset_index(name="count")
            )

            # normalize the 'count' feature
            df_eda["normalized_count"] = df_eda["count"] / data.shape[0]
            df_eda["capital_gain"] = (
                df_eda["capital_gain"].map({"No": "Capital gains? No.", "Yes": "Capital gains? Yes."})
            )
            fig = px.sunburst(
                df_eda,
                path=df_eda.drop("count", axis=1).columns[:-1],
                values=df_eda.columns[-1]
            )
            fig.update_layout(
                height=1500,
                width=1500
            )
            st.plotly_chart(fig)

    if selection == "Predict":
        st.write("#### Required Information:")

        # categorical features
        genders = sorted(set(data["gender"]))[::-1]
        gender = st.selectbox("Gender", genders)

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
                occupation=occupation,
                capital_gain=capital_gain,
                education=education,
                relationship=relationship,
                gender=gender,
                age=age,
                marital_status=marital_status
            ).to_dataframe()
            prediction = PredictPipeline().predict(record)
            mapper = {
                "<=50K": "This individual does not make more than $50,000 per year",
                ">50K": "This individual makes more than $50,000 per year"
            }
            st.subheader(prediction.replace(prediction, mapper[prediction]))


if __name__ == "__main__":
    create_app()
