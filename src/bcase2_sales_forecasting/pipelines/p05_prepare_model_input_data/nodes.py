"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def split_into_training_validation_data(
    data: pd.DataFrame, 
    parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data into training and validation sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """

    # check how many features to include
    num_feature_columns = parameters['num_feature_cols']
    if num_feature_columns == 0:

        data = data[[parameters['date_column'], parameters['target_column']]]

        # Check for if there are null values
        assert [col for col in data.columns if data[col].isnull().any()] == []
        
        # Determine the split index (as we have time series data)
        split_index = int(len(data) * parameters["validation_fraction"])

        # Split the DataFrame into train and validation
        df_train = data.iloc[:split_index]
        df_val = data.iloc[split_index:]

        df_train = pd.DataFrame({
                        'ds': df_train[parameters['date_column']],
                        'y': df_train[parameters['target_column']]
                        })

        df_val = pd.DataFrame({
                        'ds': df_val[parameters['date_column']],
                        'y': df_val[parameters['target_column']]
                        })

        return df_train, df_val
    else:
        raise NotImplementedError('The support of using the feature engineered features is not implemented yet!')
    


def prepare_full_training_data(
    data: pd.DataFrame, 
    parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data into training and validation sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """

    # check how many features to include
    num_feature_columns = parameters['num_feature_cols']
    if num_feature_columns == 0:

        data = data[[parameters['date_column'], parameters['target_column']]]

        # Check for if there are null values
        assert [col for col in data.columns if data[col].isnull().any()] == []


        df_train = pd.DataFrame({
                        'ds': data[parameters['date_column']],
                        'y': data[parameters['target_column']]
                        })

        return df_train
    else:
        raise NotImplementedError('The support of using the feature engineered features is not implemented yet!')
    

def prepare_test_data(
    test_data: pd.DataFrame, 
    parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepares Test Data.

    Args:
        tests_data: Data containing test data.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Prepared test data.
    """

    # check how many features to include
    num_feature_columns = parameters['num_feature_cols']
    if num_feature_columns == 0:

        df_test = test_data[[parameters['date_column'], parameters['target_column']]]

        # Check for if there are null values
        assert [col for col in df_test.columns if df_test[col].isnull().any()] == []

        df_test = pd.DataFrame({
                                'ds': test_data[parameters['date_column']],
                                'y': test_data[parameters['target_column']]
                                })
        return df_test
    else:
        raise NotImplementedError('The support of using the feature engineered features is not implemented yet!')