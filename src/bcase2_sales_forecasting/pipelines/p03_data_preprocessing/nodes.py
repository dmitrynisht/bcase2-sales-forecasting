from typing import Any, Dict, Tuple
import logging
from pathlib import Path
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
import hopsworks
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
import pandas as pd
from .utils import *
from colorama import Style


conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]


logger = logging.getLogger(__name__)


def preprocess_sales(
        data: pd.DataFrame, 
        parameters: Dict[str, Any]) -> pd.DataFrame:
    
    logger = logging.getLogger(__name__)

    pipeline_name = "preprocess_sales"

    # Copy the DataFrame
    sales_copy = data.copy()

    logger.info(f"'full_date' column type:\n{sales_copy[['full_date']].dtypes}")


    # Group by both 'Full_Date' (month) and 'GCK' (product), and sum the sales
    sales_copy = sales_copy.groupby([sales_copy['full_date'].dt.to_period('M'), 'gck']).sum(numeric_only=True).reset_index()
    
    # # Notebook ch3.1
    # Define a dictionary where keys are column names and values are data types
    data_types = {
        # 'full_date': 'datetime64[ns]', # already date when igested
        # 'gck': 'object', # already object when ingested
        'sales_eur': 'float32'
    }
    
    # Apply data types to the DataFrame
    for col, dtype in data_types.items():
        sales_copy[col] = sales_copy[col].astype(dtype)

    logger.info(f"The sales dataset columns convertion finished.")

    # 3.5 OUTLIERS. z-score

    # Apply the function to check normality for each product
    normality_results = sales_copy.groupby('gck')['sales_eur'].apply(check_normality)

    # Print the results
    for product, is_normal in normality_results.items():
        if is_normal:
            logger.info(f"Product {product}: ==>> Normally Distributed <<==")
        else:
            pass

    # Filter products with Normal Distribution
    is_normal_list = [product for product, is_normal in normality_results.items() if is_normal]
    sales_normal = sales_copy[sales_copy['gck'].isin(is_normal_list)]

    # Filter the DataFrame using outlier indices
    outliers_df = get_outliers(sales_normal, detect_outliers_zscore)

    # Print the DataFrame containing outliers
    logger.info(f"Outliers detected (normal):\n{outliers_df}")
    
    # Only three of the products present a Normal Distribution, and by those standards product #3 has one outlier
    # Since most of the products do not follow a Normal Distribution IQR can be helpfull do detect outliers

    # Filter the DataFrame using outlier indices
    outliers_df_iqr = get_outliers(sales_copy, detect_outliers_iqr)

    # Print the DataFrame containing outliers detected using IQR
    logger.info(f"Outliers detected (iqr):\n{outliers_df_iqr}")

    # Overall, although IQR shows a lot of outliers we will consider only the lowest value in product #1... and for product #3, 
    # since these follow a Normal Distribution, we will give a higher importance to the previous graph
    # This means that only 2 values need to be treated
    # one for product #1 in November 2018, and 
    # another for product #3 in January 2021
    
    # For product #1 (2018-11-01) we will substitute the outlier by the mean of the first 4 months
    logger.info(f"============>>")
    product_1_first_6_months_df = sales_copy.query('gck == "#1" and full_date < "2019-04-30"')
    logger.info(f"sales_copy.loc[sales_copy.query(gck == #1 and full_date < 2019-04-30)]:\n{product_1_first_6_months_df}\ntype: {type(product_1_first_6_months_df)}")
    
    # Filter the DataFrame for product #1 and the first 6 months and calculate mean
    mean_sales_product_1_first_6_months = product_1_first_6_months_df['sales_eur'].mean()

    request_prod = (sales_copy['gck'] == '#1')
    request_date = (sales_copy['full_date'] == "2018-11")
    product_1_month_11_2018_df = sales_copy[request_prod & request_date]
    logger.info(f"====================>> {product_1_month_11_2018_df.iloc[0, 0]}")
    logger.info(f"sales_copy.loc[sales_copy.query(gck == #1 and full_date == 2018-11)]:\n{product_1_month_11_2018_df}\ntype: {type(product_1_month_11_2018_df)}\nmean_sales_product_1_first_6_months: {mean_sales_product_1_first_6_months}")
    
    sales_copy.loc[sales_copy[request_prod & request_date].index, 'sales_eur'] = mean_sales_product_1_first_6_months

    # For product #3 (2021-01-01) since it follows a normal distribution we will use Z-score 
    # Calculate mean and standard deviation
    request_prod = (sales_copy['gck'] == '#3')
    request_date = (sales_copy['full_date'] == "2021-01")
    mean_sales = sales_copy[request_prod]["sales_eur"].mean()
    std_dev_sales = sales_copy[request_prod]["sales_eur"].std()

    # Fill
    sales_copy.loc[sales_copy[request_prod & request_date].index, 'sales_eur'] = round(mean_sales - 2.9 * std_dev_sales, 2)

    # Repeating the Normality Test
    normality_results2 = sales_copy.groupby('gck')['sales_eur'].apply(check_normality)

    # Print the results
    for product, is_normal in normality_results2.items():
        if is_normal:
            logger.info(f"Product {product}: ==>> Normally Distributed <<==")
        else:
            # logger.info(f"Product {product}: Not Normally Distributed")
            pass

    # Note: Product #1 follows a normal distribution after removing the outliers!!

    return sales_copy


def preprocess_markets(
        data: pd.DataFrame, 
        parameters: Dict[str, Any]) -> pd.DataFrame:
    
    logger = logging.getLogger(__name__)

    pipeline_name = "preprocess_markets"

    # Copy
    market_copy = data.copy()

    categorical_dtypes = ['object','string','category']
    market_numerical_features = market_copy.select_dtypes(exclude=categorical_dtypes+['datetime']).columns.tolist()
    market_numerical_features.remove('index')
    new_numerical_type = 'float16'

    # Apply data types to the DataFrame
    for col in market_numerical_features:
        market_copy[col] = market_copy[col].astype(new_numerical_type)

    logger.info(f"The market dataset {len(market_numerical_features)} columns converted to {new_numerical_type}. Conversion finished.")

    return market_copy


def market_merge_german_gdp(
        market_data: pd.DataFrame,
        gdp_data: pd.DataFrame, 
        parameters: Dict[str, Any]) -> pd.DataFrame:
    
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

    return market_copy
