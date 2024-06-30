import pandas as pd
import numpy as np
from typing import Any, Dict
from scipy.stats import shapiro
import logging


logger = logging.getLogger(__name__)


def check_normality(data: pd.Series):
    """Function to check normality using Shapiro-Wilk test
    """
    _, p_value = shapiro(data)
    
    return p_value > 0.05  # Null hypothesis: data is normally distributed if p_value > 0.05


def detect_outliers_zscore(data: pd.Series):
    """Function to identify outliers using z-score
    """
    
    threshold = 3
    mean = np.mean(data)
    std_dev = np.std(data)
    z_scores = [(x - mean) / std_dev for x in data]
    
    return np.abs(z_scores) > threshold


def detect_outliers_iqr(data: pd.Series):
    """Function to identify outliers using IQR
    """

    quartile_1, quartile_3 = np.percentile(data, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (1.5 * iqr)
    upper_bound = quartile_3 + (1.5 * iqr)
    
    return (data < lower_bound) | (data > upper_bound)


def get_outliers(data: pd.DataFrame, detect_outliers_funcrion: Any):
    """Function returns filtered data. Data is being filtered according to detect_outliers_funcrion passed
    as second argument.
    Expected names for detect_outliers_funcrion: 
    - detect_outliers_zscore
    - detect_outliers_iqr
    """

    # Create an empty list to store outlier indices
    outlier_indices = []

    # Iterate over groups
    for group_name, group_data in data.groupby('gck')['sales_eur']:
        # Detect outliers for the current group
        outliers_group = group_data.pipe(detect_outliers_funcrion)
        # Append outlier indices to the list
        outlier_indices.extend(group_data[outliers_group].index)

    # Filter the DataFrame using outlier indices
    outliers_df_iqr = data.loc[outlier_indices]

    return outliers_df_iqr