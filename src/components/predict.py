import sys
import numpy as np
import pandas as pd

from catboost import Pool
from sklearn.metrics import roc_auc_score

from src.logger import logging
from src.exception import CustomException
from src.utils import load_artifact
from src.components.ingest import DataIngestion
from src.components.process import DataProcessor


class Predict:
    def __init__(self):
        self.model = load_artifact(r"./artifacts/catboost_clf_all_features")

    def evaluate(self, feature_matrix: pd.DataFrame, target_vector: pd.Series):
        try:
            pool_data = Pool(
                feature_matrix,
                target_vector,
                cat_features=feature_matrix.select_dtypes("O").columns.tolist()
            )
            prediction_vector: np.ndarray = self.model.predict(pool_data)
            pos_class_pred_proba: np.ndarray = self.model.predict_proba(pool_data)[:, 1]
            accuracy: float = np.mean(target_vector.values == prediction_vector)
            auc_score: float = roc_auc_score(target_vector, pos_class_pred_proba)
            logging.info(
                "AUC Score: %s, Accuracy: %s",
                np.round(auc_score, 2),
                np.round(accuracy, 2)
            )
            return auc_score, accuracy
        except Exception as err:
            raise CustomException(err, sys) from err


if __name__ == "__main__":
    raw_data_path = DataIngestion().ingest_data()
    _, x_test, _, y_test = DataProcessor(raw_data_path).process_data()
    _, _ = Predict().evaluate(x_test, y_test)
