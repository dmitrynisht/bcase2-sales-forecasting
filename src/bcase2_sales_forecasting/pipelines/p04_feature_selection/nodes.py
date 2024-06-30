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


def market_features_selection(
        market_data: pd.DataFrame,
        parameters: Dict[str, Any]) -> pd.DataFrame:
    
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

    return market_copy

def compute_sales_lag_features(
        sales_data: pd.DataFrame,
        market_data: pd.DataFrame,
        parameters: Dict[str, Any]) -> pd.DataFrame:
    # Only return sales of product specified in parameters
    # fallback to #1 if not specified
    if parameters.get("target_product"):
        product_code = parameters.get("target_product")
    else:
        logger.warn("No target code specified in parameters. Defaulting to #1")
        product_code = "#1"
    
    product_sales = sales_data[sales_data["gck"] == product_code]

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

    # Loop through each product group
    sales_lag = find_lag(market_data, product_sales, mkt_lagged_datasets)
    sales_lag.set_index("full_date", inplace=True)
    sales_lag.drop("gck", axis=1, inplace=True)

    add_sales_lags(sales_lag)
    sales_lag.dropna(inplace=True, axis=1)

    sales_lag = sales_lag[list(set(['sales_eur'] + get_highly_correlated_features(sales_lag) + get_top_10_features(sales_lag)))]
    sales_lag.dropna(inplace=True)
    sales_lag.reset_index(inplace=True)

    # Convert the date_column to datetime format
    sales_lag[parameters['date_column']] = pd.to_datetime(sales_lag[parameters['date_column']], format='%d-%m-%Y')

    # Reformat the date_column to the desired format
    sales_lag[parameters['date_column']] = sales_lag[parameters['date_column']].dt.strftime('%Y-%m-%d')
 
    return sales_lag