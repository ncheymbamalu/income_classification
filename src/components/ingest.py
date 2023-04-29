from typing import Dict, Any
import os
import sys

from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import load_artifact, save_artifact


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw_data.parquet")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.params: Dict[str, Any] = load_artifact(r"./conf/parameters.yml")

    def ingest_data(self):
        try:
            logging.info("Data ingestion initiated")
            raw_data = load_artifact(r"./data/income_evaluation.parquet")

            logging.info(
                "Saving the raw data to %s",
                os.path.join(os.getcwd(), os.path.dirname(self.ingestion_config.raw_data_path))
            )
            save_artifact(raw_data, self.ingestion_config.raw_data_path)
            logging.info("Data ingestion complete")
            return self.ingestion_config.raw_data_path
        except Exception as err:
            raise CustomException(err, sys) from err


if __name__ == "__main__":
    _ = DataIngestion().ingest_data()
