from typing import Any, Dict, Tuple
import logging
from pathlib import Path
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
import hopsworks
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
import pandas as pd
import numpy as np
from .utils import *

conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]


logger = logging.getLogger(__name__)


def preprocess_sales(
        data: pd.DataFrame, 
        parameters: Dict[str, Any],
        dummy_value) -> pd.DataFrame:
    
    logger = logging.getLogger(__name__)

    pipeline_name = "preprocess_sales"

    # Copy the DataFrame
    sales_copy = data.copy()

    # Convert 'Full_Date' column to datetime
    sales_copy['full_date'] = pd.to_datetime(sales_copy['full_date'], dayfirst=True)

    # Group by both 'Full_Date' (month) and 'GCK' (product), and sum the sales
    sales_copy = sales_copy.groupby([sales_copy['full_date'].dt.to_period('M'), 'gck']).sum(numeric_only=True).reset_index()

    # # Notebook ch3.1
    # Define a dictionary where keys are column names and values are data types
    data_types = {
        # 'full_date': 'datetime64[ns]',
        # 'gck': 'object',
        'sales_eur': 'float32'
    }
    
    # Apply data types to the DataFrame
    for col, dtype in data_types.items():
        sales_copy[col] = sales_copy[col].astype(dtype)

    logger.info(f"The sales dataset columns convertion finished.")

    pass

    # Set True/False whenever debug needed/or not
    if parameters["debug_output"][pipeline_name]:
        # Printing something from dataframe (usually columns)
        # dummy_value is for checking pipelines sequence
        f_verbose = True
        debug_on_success_(sales_copy, dummy_value, pipeline_name, f_verbose)

    return sales_copy, dummy_value


def preprocess_markets(
        data: pd.DataFrame, 
        parameters: Dict[str, Any],
        dummy_value) -> pd.DataFrame:
    
    logger = logging.getLogger(__name__)

    pipeline_name = "preprocess_markets"

    # Copy
    market_copy = data.copy()

    categorical_dtypes = ['object','string','category']
    market_numerical_features = market_copy.select_dtypes(exclude=categorical_dtypes+['datetime']).columns.tolist()
    market_numerical_features.remove('index')
    # print("print to remove, markets preprocessing !".upper(), "list of columns:", market_numerical_features)
    new_numerical_type = 'float16'

    # Apply data types to the DataFrame
    for col in market_numerical_features:
        market_copy[col] = market_copy[col].astype(new_numerical_type)

    logger.info(f"The market dataset {len(market_numerical_features)} columns converted to {new_numerical_type}. Conversion finished.")

    pass

    # Set True/False whenever debug needed/or not
    if parameters["debug_output"][pipeline_name]:
        # Printing something from dataframe (usually columns)
        # dummy_value is for checking pipelines sequence
        f_verbose = True
        debug_on_success_(market_copy, dummy_value, pipeline_name, f_verbose)

    return market_copy, dummy_value


def market_merge_german_gdp(
        market_data: pd.DataFrame,
        gdp_data: pd.DataFrame, 
        parameters: Dict[str, Any],
        dummy_value) -> pd.DataFrame:
    
    logger = logging.getLogger(__name__)

    pipeline_name = "market_merge_german_gdp"

    logger.info(f"{pipeline_name} / {'market features selection'}")

    # Copy
    market_copy = market_data.copy()
    gdp_copy = gdp_data.copy()

    # Convert the 'DATE' column to datetime format and set it as the index
    gdp_copy.set_index('month_year', inplace=True)

    #the .resample() method is applied to the index column of the DataFrame, which must be a datetime-like index
    # Resample the data to monthly frequency and forward fill missing values
    gdp_monthly = pd.DataFrame(gdp_copy.resample('MS').ffill()['gdp'] / 3)

    # Merge the datasets on the 'month_year' column
    market_copy = market_copy.merge(gdp_monthly.rename(columns={'gdp': 'german_gdp'}), on='month_year', how='left')

    logger.info(f"The MARKET data merged with GERMAN GDP. Processed market contains {len(market_copy.columns)} columns.")

    pass

    # Set True/False whenever debug needed/or not
    if parameters["debug_output"][pipeline_name]:
        # Printing something from dataframe (usually columns)
        # dummy_value is for checking pipelines sequence
        f_verbose = True
        debug_on_success_(market_copy, dummy_value, pipeline_name, f_verbose)

    return market_copy, dummy_value
