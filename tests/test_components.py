import pytest
from src.utils import load_artifact
from src.components.ingest import DataIngestion
from src.components.process import DataProcessor
from src.components.predict import Predict


@pytest.fixture
def get_raw_data_path():
    path = DataIngestion().ingest_data()
    return path


def test_ingest(get_raw_data_path):
    df_raw = load_artifact(get_raw_data_path)
    assert isinstance(get_raw_data_path, str)
    assert df_raw.shape[0] > 0
    assert df_raw.shape[1] > 0


def test_process(get_raw_data_path):
    x_train, x_test, _, _ = DataProcessor(get_raw_data_path).process_data()
    assert x_train.isna().sum().sum() == 0
    assert x_test.isna().sum().sum() == 0


def test_predict(get_raw_data_path):
    _, x_test, _, y_test = DataProcessor(get_raw_data_path).process_data()
    test_auc, test_accuracy = Predict().evaluate(x_test, y_test)
    baseline_auc = 0.5
    baseline_accuracy = y_test.value_counts("normalize").values[0]
    assert test_auc > baseline_auc
    assert test_accuracy > baseline_accuracy
