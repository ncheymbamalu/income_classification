import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_artifact


class CustomData:
    def __init__(
            self,
            occupation: str,
            capital_gain: str,
            education: str,
            relationship: str,
            gender: str,
            age: int,
            marital_status: str
    ):
        self.occupation = occupation
        self.capital_gain = capital_gain
        self.education = education
        self.relationship = relationship
        self.gender = gender
        self.age = age
        self.marital_status = marital_status

    def to_dataframe(self):
        try:
            return pd.DataFrame(
                {
                    "occupation": [self.occupation],
                    "capital_gain": [self.capital_gain],
                    "education": [self.education],
                    "relationship": [self.relationship],
                    "gender": [self.gender],
                    "age": [self.age],
                    "marital_status": [self.marital_status]
                }
            )
        except Exception as err:
            raise CustomException(err, sys) from err


class PredictPipeline:
    def __init__(self):
        self.model = load_artifact(r"./artifacts/catboost_clf_rel_features")

    def predict(self, record: pd.DataFrame):
        try:
            prediction = self.model.predict(record)[0]
            return prediction
        except Exception as err:
            raise CustomException(err, sys) from err
