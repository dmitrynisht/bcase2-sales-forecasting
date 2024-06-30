from typing import Any, Dict, Tuple
import logging
from pathlib import Path
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
import pandas as pd
import numpy as np
from .utils import *

conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]


logger = logging.getLogger(__name__)


def sales_features_selection(
        sales_data: pd.DataFrame,
        parameters: Dict[str, Any],
        dummy_value) -> pd.DataFrame:
    
    logger = logging.getLogger(__name__)

    pipeline_name = "sales_features_selection"

    logger.info(f"{pipeline_name}")

    # Copy
    sales_copy = sales_data.copy()

    pass

    # Set True/False whenever debug needed/or not
    if parameters["debug_output"][pipeline_name]:
        # Printing something from dataframe (usually columns)
        # dummy_value is for checking pipelines sequence
        f_verbose = True
        debug_on_success_(sales_copy, dummy_value, pipeline_name, f_verbose)

    return sales_copy


def market_features_selection(
        market_data: pd.DataFrame,
        parameters: Dict[str, Any],
        dummy_value) -> pd.DataFrame:
    
    logger = logging.getLogger(__name__)

    pipeline_name = "market_features_selection"

    logger.info(f"{pipeline_name}")

    # Copy
    market_copy = market_data.copy()

    # Removing correlated features

    # Select only numerical features
    numerical_features = market_copy.select_dtypes(include=[np.number])

    # Calculate correlation matrix
    correlation_matrix = numerical_features.corr().abs()

    # Create a mask to select the upper triangle of correlation matrix
    mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)

    # Select upper triangle of correlation matrix using the mask
    upper_triangle = correlation_matrix.where(mask)

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
    logger.info(f"{pipeline_name.upper()}. features to drop:")
    logger.info(f"{to_drop}")

    # Drop highly correlated features
    market_copy = market_copy.drop(columns=to_drop)
    
    logger.info(f"The {'market dataset'.upper()} processing finished.")

    pass

    # Set True/False whenever debug needed/or not
    if parameters["debug_output"][pipeline_name]:
        # Printing something from dataframe (usually columns)
        # dummy_value is for checking pipelines sequence
        f_verbose = True
        debug_on_success_(market_copy, dummy_value, pipeline_name, f_verbose)

    return market_copy, dummy_value

def compute_sales_lag_features(
        sales_data: pd.DataFrame,
        market_data: pd.DataFrame,
        parameters: Dict[str, Any]) -> pd.DataFrame:
    # Creating a dataframe for each product sales
    product_sales_map = {}
    for product_name in sales_data["gck"].unique():
        # Modify product name to ensure it's a valid variable name
        valid_product_name = re.sub(r'\W+', '_#', product_name)

        # Dynamically create variable name based on product name
        product_sales_map[f"sales{valid_product_name}"] = sales_data[sales_data["gck"] == product_name]

    # Set 'Month Year' as index
    market_data['month_year'] = pd.to_datetime(market_data['month_year'])
    market_data.set_index('month_year', inplace=True)

    # Define lag periods
    lag_periods = range(1, 13)

    # Create a dictionary to store lagged datasets
    mkt_lagged_datasets = {}

    # Generate lagged datasets
    for lag in lag_periods:
        start_date = pd.to_datetime('2018-09-01') - pd.DateOffset(months=lag)
        end_date = pd.to_datetime('2022-03-01') - pd.DateOffset(months=lag)
        mkt_lagged_datasets[f'market_lag{lag}'] = market_data[start_date:end_date].reset_index(drop=True)
    
    sales_lag_per_product = {}

    # Loop through each product group
    for product_code in sales_data["gck"].unique():
        sales_lag = find_lag(product_code, market_data, product_sales_map, mkt_lagged_datasets)
        sales_lag.set_index("full_date", inplace=True)
        sales_lag.drop("gck", axis=1, inplace=True)

        sales_lag_per_product[product_code] = sales_lag

    lag_datasets = sales_lag_per_product.values()

    # Apply the function to each lag dataset
    for lag_data in lag_datasets:
        add_sales_lags(lag_data)

    # Only return sales of product specified in parameters
    # fallback to #1 if not specified
    product_code = parameters.get(product_code) or "#1"
    sales_p1 = sales_lag_per_product[product_code]

    #TODO: Create specific nodes in pipeline to get corr and top feats
    sales_p1 = sales_p1[list(set(['sales_eur'] + get_highly_correlated_features(sales_p1) + get_top_10_features(sales_p1)))]
    sales_p1.dropna(inplace=True)
    sales_p1.reset_index(inplace=True)

    return [sales_p1]