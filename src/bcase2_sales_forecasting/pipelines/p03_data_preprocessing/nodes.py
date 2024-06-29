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
        parameters: Dict[str, Any],
        dummy_value) -> pd.DataFrame:
    
    logger = logging.getLogger(__name__)

    pipeline_name = "preprocess_sales"

    # Copy the DataFrame
    sales_copy = data.copy()

    logger.info(f"'full_date' column type:\n{sales_copy[['full_date']].dtypes}")

    # # Convert 'Full_Date' column to datetime already done while ingesting
    # sales_copy['full_date'] = pd.to_datetime(sales_copy['full_date'], dayfirst=True)

    # HERE IS A PROBLEM: I DON'T KNOW WHY, BUT IN JUPYTER FILE THE OUTPUT OF THIS LINE OF CODE GENERATES 620 ROWS
    # THE OUTPUT OF THIS LINE OF CODE IN THIS PIPELINE IS 790 ROWS
    # AND PERHAPS BECAUSE OF THAT IN THE normality_results WE DON'T HAVE ANY NORMALLY DISTRIBUTED GROUP
    # BUT IN JUPYTER FILE WE HAVE INITIALLY GROUPS 3, 5, 6 NORMALLY DISTRIBUTED, AND AFTER REMOVING OUTLIRES ALSO GROUP 1
    # WE ARE LIVING AS IS BUT TO BE INVESTIGATED WHY THE PIPELINE IS CORRUPTING THE DATA

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
            logger.info(f"Product {product}: {Style.BRIGHT}Normally Distributed{Style.RESET_ALL}")
        else:
            logger.info(f"Product {product}: Not Normally Distributed")

    logger.info(f"normality results:\n{normality_results}")

    # Filter products with Normal Distribution
    # sales_normal = sales_copy.query('GCK == "#3" | GCK == "#5" | GCK == "#6"') # these were expected to be normally distributed
    is_normal_list = [product for product, is_normal in normality_results.items() if is_normal]
    sales_normal = sales_copy[sales_copy['gck'].isin(is_normal_list)]

    # Create an empty list to store outlier indices
    outlier_indices = []

    # Iterate over groups
    for group_name, group_data in sales_normal.groupby('gck')['sales_eur']:
        # Detect outliers for the current group
        outliers_group = group_data.pipe(detect_outliers_zscore)
        # Append outlier indices to the list
        outlier_indices.extend(group_data[outliers_group].index)
        
    # Filter the DataFrame using outlier indices
    outliers_df = sales_normal.loc[outlier_indices]

    # Print the DataFrame containing outliers
    logger.info(f"Outliers detected:\n{outliers_df}")
    # Only three of the products present a Normal Distribution, and by those standards product #3 has one outlier
    # Since most of the products do not follow a Normal Distribution IQR can be helpfull do detect outliers

    # Create an empty list to store outlier indices
    outlier_indices = []

    # Iterate over groups
    for group_name, group_data in sales_copy.groupby('gck')['sales_eur']:
        # Detect outliers for the current group
        outliers_group = group_data.pipe(detect_outliers_iqr)
        # Append outlier indices to the list
        outlier_indices.extend(group_data[outliers_group].index)

    # Filter the DataFrame using outlier indices
    outliers_df_iqr = sales_copy.loc[outlier_indices]

    # Print the DataFrame containing outliers detected using IQR
    logger.info(f"Outliers detected:\n{outliers_df_iqr}")
    # Overall, although IQR shows a lot of outliers we will consider only the lowest value in product #1... and for product #3 and #5, since these follow a Normal Distribution, we will give a higher importance to the previous graph
    # This means that only 2 values need to be treated one for product #1 in November 2018, and another for product #3 in January 2021
    # 
    # ###############################################3
    # # For product #1 (2018-11-01) we will substitute the outlier by the mean of the first 4 months

    # # Filter the DataFrame for product #1 and the first 6 months
    # mean_sales_product_1_first_6_months = sales_copy.query('GCK == "#1" and Full_Date < "2019-04-30"')['Sales €'].mean()

    # sales_copy.loc[sales_copy.query('GCK == "#1" and Full_Date == "2018-11-01"').index, 'Sales €'] = mean_sales_product_1_first_6_months

    # # For product #3 (2021-01-01) since it follows a normal distribution we will use Z-score 
    # # Calculate mean and standard deviation
    # mean_sales = sales_copy[sales_copy['GCK'] == "#3"]["Sales €"].mean()
    # std_dev_sales = sales_copy[sales_copy['GCK'] == "#3"]["Sales €"].std()

    # # Fill
    # sales_copy.loc[sales_copy.query('GCK == "#3" and Full_Date == "2021-01-01"').index, 'Sales €'] = round(mean_sales - 2.9 * std_dev_sales,2)

    # # Repeating the Normality Test
    # normality_results2 = sales_copy.groupby('GCK')['Sales €'].apply(check_normality)

    # # Print the results
    # for product, is_normal in normality_results2.items():
    #     if is_normal:
    #         print(f"Product {product}: {Style.BRIGHT}Normally Distributed{Style.RESET_ALL}")
    #     else:
    #         print(f"Product {product}: Not Normally Distributed")

    # # Note: Product #1 follows a normal distribution after removing the outliers!!
    # ################################################
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
