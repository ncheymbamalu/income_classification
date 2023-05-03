from typing import Tuple, List, Dict, Any
import os
import sys
import pickle
import yaml
import numpy as np
import pandas as pd
import catboost
import streamlit as st

from yaml import SafeLoader
from ydata_profiling import ProfileReport
from sklearn.feature_selection import mutual_info_classif
from mrmr import mrmr_classif
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
            return pd.read_parquet(path)
        if path.split(".")[-1] == "yml":
            return yaml.load(open(path), Loader=SafeLoader)
        if path.split(".")[-1] == "pkl":
            return pickle.load(open(path, "rb"))
        if "catboost_clf" in path:
            return CatBoostClassifier().load_model(path)
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
        if isinstance(obj, catboost.core.CatBoostClassifier):
            obj.save_model(path)
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


@st.cache_data
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
            .drop_duplicates(keep="first")
            .copy(deep=True)
        )
        return df_preprocessed
    except Exception as err:
        raise CustomException(err, sys) from err


@st.cache_resource
def create_profile_report(data: pd.DataFrame) -> Any:
    """
    Creates a ydata-profiling report for data

    Args:
        data: DataFrame

    Returns:
        profile: ydata-profiling report
    """
    try:
        profile = ProfileReport(data, explorative=True, dark_mode=True)
        return profile
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

        # encode the categorical features
        ctoi, ctov, vtoi, itoc = {}, {}, {}, {}
        for col in nominal_cols + ordinal_cols:
            if col in nominal_cols:
                categories = x_train[col].value_counts("normalize").index.tolist()
                normalized_counts = x_train[col].value_counts("normalize").values.tolist()
                indices = np.arange(len(categories)).tolist()[::-1]
                ctov[col] = dict(zip(categories, normalized_counts))
                vtoi[col] = dict(zip(normalized_counts, indices))
                itoc[col] = dict(zip(indices, categories))
                x_train[col] = x_train[col].map(ctov[col]).map(vtoi[col])
                x_test[col] = x_test[col].map(ctov[col]).map(vtoi[col])
            else:
                categories = params["ordinal"]["categories"][col]
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


def get_informative_features(x_train: pd.DataFrame, y_train: pd.Series) -> List[str]:
    """
    Returns a list containing the most informative features

    Args:
        x_train: Train set feature matrix
        y_train: Train set target vector

    Returns:
        informative_features: List containing the most informative features
    """
    try:
        df_train = pd.concat([x_train, y_train], axis=1).copy(deep=True)
        params = load_artifact(r"./conf/parameters.yml")

        # encode the target variable
        target = params["target"]
        target_classes = sorted(set(df_train[target]))
        target_encoder = dict(zip(target_classes, [0, 1]))
        df_train[target] = df_train[target].map(target_encoder)

        # a dictionary that maps each feature to its mutual info score
        mutual_info = {}
        numeric_cols = params["numeric_features"]
        nominal_cols = params["nominal_features"]
        ordinal_cols = params["ordinal"]["features"]
        for col in numeric_cols + nominal_cols + ordinal_cols:
            # standardize the numeric features
            if col in numeric_cols:
                train_mean, train_std = df_train[col].mean(), df_train[col].std()
                df_train[col] = (df_train[col] - train_mean) / train_std
                score = mutual_info_classif(
                    df_train[[col]],
                    df_train[target],
                    random_state=params["random_state"]
                )[0]
                mutual_info[col] = score
            # label encode and standardize the nominal features
            elif col in nominal_cols:
                # categories = df_train.groupby(col)[target].mean().index.tolist()
                # values = df_train.groupby(col)[target].mean().values.tolist()
                # ctov = dict(zip(categories, values))
                # df_train[col] = df_train[col].map(ctov)
                categories = df_train[col].value_counts("normalize").index.tolist()
                normalized_counts = df_train[col].value_counts("normalize").values.tolist()
                indices = np.arange(len(normalized_counts)).tolist()[::-1]
                ctov = dict(zip(categories, normalized_counts))
                vtoi = dict(zip(normalized_counts, indices))
                df_train[col] = df_train[col].map(ctov).map(vtoi)
                train_mean, train_std = df_train[col].mean(), df_train[col].std()
                df_train[col] = (df_train[col] - train_mean) / train_std
                score = mutual_info_classif(
                    df_train[[col]],
                    df_train[target],
                    random_state=params["random_state"]
                )
                mutual_info[col] = score[0]
            else:
                # ordinal encode and standardize the ordinal features
                categories = params["ordinal"]["categories"][col]
                indices = range(len(categories))
                ctoi = dict(zip(categories, indices))
                df_train[col] = df_train[col].map(ctoi)
                train_mean, train_std = df_train[col].mean(), df_train[col].std()
                df_train[col] = (df_train[col] - train_mean) / train_std
                score = mutual_info_classif(
                    df_train[[col]],
                    df_train[target],
                    random_state=params["random_state"]
                )
                mutual_info[col] = score[0]

        # specify a threshold mutual information score
        threshold = np.mean(list(mutual_info.values()))

        # remove the features whose mutual info score is less than the threshold
        mutual_info: dict = {
            col: score.round(4)
            for (col, score) in mutual_info.items()
            if score > threshold
        }

        # most informative features via mutual information
        mutual_info_features = list(mutual_info.keys())

        # most informative features via maximum relevancy, minimum redundancy
        mrmr_features: list = mrmr_classif(
            X=df_train.drop(target, axis=1),
            y=df_train[target],
            K=len(mutual_info),
            relevance="f",
            redundancy="c"
        )
        informative_features = list(set(mutual_info_features + mrmr_features))
        return informative_features
    except Exception as err:
        raise CustomException(err, sys) from err
