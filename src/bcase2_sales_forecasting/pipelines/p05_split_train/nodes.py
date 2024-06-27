"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def split_data(
    data: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into training and validation sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """

    # Check for if there are null values
    assert [col for col in data.columns if data[col].isnull().any()] == []

    # Determine the split index (as we have time series data)
    split_index = int(len(data) * parameters["validation_fraction"])

    # Split the DataFrame into train and validation
    df_train = data.iloc[:split_index]
    df_val = data.iloc[split_index:]

    # Targets
    y_train = df_train[parameters["target_column"]]
    y_val = df_val[parameters["target_column"]]

    # Features
    X_train = df_train.drop(columns=parameters["target_column"], axis=1)
    X_val = df_val.drop(columns=parameters["target_column"], axis=1)

    return X_train, X_val, y_train, y_val
