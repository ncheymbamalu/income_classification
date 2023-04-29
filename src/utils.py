"""Module docstring"""
from typing import Tuple, Dict, Any
import os
import sys
import pickle
import yaml
import numpy as np
import pandas as pd

from yaml import SafeLoader
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

from src.exception import CustomException


def load_artifact(path: str) -> Any:
    """
    Reads in an artifact from path as a Python object

    Args:
        path: The artifact's file path

    Returns:
        obj: Python object
    """
    try:
        if path.split(".")[-1] == "parquet":
            obj = pd.read_parquet(path)
        elif path.split(".")[-1] == "yml":
            obj = yaml.load(open(path), Loader=SafeLoader)
        elif path.split(".")[-1] == "pkl":
            obj = pickle.load(open(path, "rb"))
        elif "catboost_clf" in path:
            obj = CatBoostClassifier().load_model(path)
        return obj
    except Exception as err:
        raise CustomException(err, sys) from err


def save_artifact(obj: Any, path: str) -> None:
    """
    Writes obj to path

    Args:
        obj: Python object
        path: File path obj is written to
    """
    try:
        directory: str = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        if path.split(".")[-1] == "pkl":
            pickle.dump(obj, open(path, "wb"))
        if path.split(".")[-1] == "parquet":
            obj.to_parquet(path)
    except Exception as err:
        raise CustomException(err, sys) from err


def remove_whitespace(data: pd.DataFrame) -> pd.DataFrame:
    """
    Removes the white space from all categorical
    column entries
    """
    try:
        for col in data.select_dtypes("O").columns:
            data[col] = data[col].str.replace(" ", "")
        return data
    except Exception as err:
        raise CustomException(err, sys) from err


def preprocess_data(path: str) -> pd.DataFrame:
    """
    Returns a pre-processed DataFrame

    Args:
        path: Raw data's file path

    Returns:
        df_preprocessed: Pre-processed DataFrame
    """
    try:
        data: pd.DataFrame = load_artifact(path)
        data.columns = [
            col.replace(" ", "").replace("-", "_").replace("sex", "gender")
            for col in data.columns
        ]
        df_preprocessed = (
            data
            .pipe(remove_whitespace)
            .replace("?", np.nan)
            .assign(
                capital_gain=(
                    data["capital_gain"].apply(lambda x: "No" if x == 0 else "Yes").astype("object")
                ),
                capital_loss=(
                    data["capital_loss"].apply(lambda x: "No" if x == 0 else "Yes").astype("object")
                )
            )
            .drop(["education_num"], axis=1)
            .copy(deep=True)
        )
        return df_preprocessed
    except Exception as err:
        raise CustomException(err, sys) from err


def impute_features(
        data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns imputed train and test set features and targets

    Args:
        data: Pre-processed data with missing values

    Returns:
        x_train: Train set feature matrix with imputed values
        x_test: Test set features matrix with imputed values
        y_train: Train set target vector
        y_test: Test set target vector
    """
    try:
        # read in './conf/parameters.yml'
        params: Dict[str, Any] = load_artifact(r"./conf/parameters.yml")

        # split the data into train and test sets
        target: str = params["target"]
        train_set, test_set = train_test_split(
            data,
            test_size=params["test_size"],
            stratify=data[target],
            shuffle=True,
            random_state=params["random_state"]
        )
        x_train = train_set.drop(target, axis=1).copy(deep=True)
        y_train = train_set[target].copy(deep=True)
        x_test = test_set.drop(target, axis=1).copy(deep=True)
        y_test = test_set[target].copy(deep=True)

        # specify the numeric and categorical (nominal and ordinal) features
        numeric_cols: list = params["numeric_features"]
        nominal_cols: list = params["nominal_features"]
        ordinal_cols: list = params["ordinal"]["features"]

        # a list to store features that have missing values
        null_cols = [
            col
            for col in numeric_cols + nominal_cols + ordinal_cols
            if x_train[col].isna().sum() > 0
        ]

        # label encode the categorical features
        ctoi, itoc = {}, {}
        for col in nominal_cols + ordinal_cols:
            categories = sorted(set(x_train[col].dropna()))
            indices = range(len(categories))
            ctoi[col] = dict(zip(categories, indices))
            itoc[col] = dict(zip(indices, categories))
            x_train[col] = x_train[col].map(ctoi[col])
            x_test[col] = x_test[col].map(ctoi[col])

        # read in './artifacts/imputer.pkl'
        imputer = load_artifact(r"./artifacts/imputer.pkl")

        # impute the train and test set features that have missing values
        x_train = pd.DataFrame(
            imputer.transform(x_train),
            columns=x_train.columns.tolist(),
            index=x_train.index.tolist()
        )
        x_test = pd.DataFrame(
            imputer.transform(x_test),
            columns=x_test.columns.tolist(),
            index=x_test.index.tolist()
        )

        # map each categorical feature back to its original categories
        for col in nominal_cols + ordinal_cols:
            if col in null_cols:
                x_train[col] = np.abs(x_train[col]).round().astype(int).map(itoc[col])
                x_test[col] = np.abs(x_test[col]).round().astype(int).map(itoc[col])
            else:
                x_train[col] = x_train[col].astype(int).map(itoc[col])
                x_test[col] = x_test[col].astype(int).map(itoc[col])
        return x_train, x_test, y_train, y_test
    except Exception as err:
        raise CustomException(err, sys) from err