import pandas as pd
import numpy as np
from typing import Any, Dict
from scipy.stats import shapiro


def debug_on_success_(data: pd.DataFrame, dummy_value: int, pipeline_name: str = "", f_verbose: bool = False) -> None:
    
    # Print columns
    if f_verbose:
        print(data.dtypes)

    # dummy_value is for checking pipelines sequence
    dummy_value.append(dummy_value[-1] + 1) 
    print(f"pipeline {pipeline_name} succeed !; f_verbose={f_verbose};", dummy_value)

    return


def check_normality(data: pd.Series):
    """Function to check normality using Shapiro-Wilk test
    """
    _, p_value = shapiro(data)
    
    return p_value > 0.05  # Null hypothesis: data is normally distributed if p_value > 0.05


def detect_outliers_zscore(data):
    """Function to identify outliers using z-score
    """
    print(type(data))
    threshold = 3
    mean = np.mean(data)
    std_dev = np.std(data)
    z_scores = [(x - mean) / std_dev for x in data]
    
    return np.abs(z_scores) > threshold