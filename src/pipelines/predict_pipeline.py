import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_artifact


class CustomData:
    def __init__(
            self,
            education: str,
            occupation: str,
            relationship: str,
            marital_status: str,
            capital_gain: str,
            age: int
    ):
        self.education = education
        self.occupation = occupation
        self.relationship = relationship
        self.marital_status = marital_status
        self.capital_gain = capital_gain
        self.age = age

    def to_dataframe(self):
        try:
            return pd.DataFrame(
                {
                    "education": [self.education],
                    "occupation": [self.occupation],
                    "relationship": [self.relationship],
                    "marital_status": [self.marital_status],
                    "capital_gain": [self.capital_gain],
                    "age": [self.age]
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
