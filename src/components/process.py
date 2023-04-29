import sys

from src.logger import logging
from src.exception import CustomException
from src.utils import preprocess_data, impute_features
from src.components.ingest import DataIngestion


class DataProcessor:
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path

    def process_data(self):
        try:
            logging.info("Data processing initiated")
            preprocessed_data = preprocess_data(self.raw_data_path)
            x_train, x_test, y_train, y_test = impute_features(preprocessed_data)
            logging.info("Data processing completed")
            return x_train, x_test, y_train, y_test
        except Exception as err:
            raise CustomException(err, sys) from err


if __name__ == "__main__":
    RAW_DATA_PATH = DataIngestion().ingest_data()
    _, _, _, _ = DataProcessor(RAW_DATA_PATH).process_data()
