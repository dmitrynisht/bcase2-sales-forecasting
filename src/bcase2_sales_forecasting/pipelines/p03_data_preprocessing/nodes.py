from typing import Any, Dict, Tuple
import logging
from pathlib import Path
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
import hopsworks
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
import pandas as pd
from .utils import *

conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]


logger = logging.getLogger(__name__)

    
# def build_expectation_suite(expectation_suite_name: str, feature_group: str) -> ExpectationSuite:
#     """
#     Builder used to retrieve an instance of the validation expectation suite.
    
#     Args:
#         expectation_suite_name (str): A dictionary with the feature group name and the respective version.
#         feature_group (str): Feature group used to construct the expectations.
            
#     Returns:
#         ExpectationSuite: A dictionary containing all the expectations for this particular feature group.
#     """
        
#     expectation_suite_bank = ExpectationSuite(
#         expectation_suite_name=expectation_suite_name
#     )

#     # numerical features
#     if feature_group == 'sales_numerical_features':
        
#         expectation_suite_bank.add_expectation(
#             ExpectationConfiguration(
#                 expectation_type="expect_table_column_count_to_equal",
#                 kwargs={"value": 3},
#             )
#         )

#         expectation_suite_bank.add_expectation(
#             ExpectationConfiguration(
#                 expectation_type="expect_column_values_to_be_of_type",
#                 kwargs={"column": "sales_€", "type_": "float64"},
#             )
#         )

#         pass

#     if feature_group == "market_total_features":

#         expected_column_names = market_columns_list_()

#         # # to improve, we don't need unique values within columns
#         # expectation_suite_bank.add_expectation(
#         #     ExpectationConfiguration(
#         #         # expectation_type="expect_table_columns_to_match_ordered_list",
#         #         # kwargs={"column_list": expected_column_names},
                
#         #         expectation_type="expect_column_values_to_be_unique",
#         #         kwargs={
#         #                 "column_map": {
#         #                     "numeric_columns": expected_column_names[2:],
#         #                     # "categorical_columns": expected_column_names[1:]
#         #                 }
#         #         }
#         #     )
#         # )

#         # expectation_suite_bank.add_expectation(
#         #     ExpectationConfiguration(
#         #         expectation_type="expect_table_column_count_to_equal",
#         #         kwargs={"value": 49},
#         #     )
#         # )

#         # # this intentionally left as example
#         # expectation_suite_bank.add_expectation(
#         #     ExpectationConfiguration(
#         #         expectation_type="expect_column_distinct_values_to_be_in_set",
#         #         kwargs={"column": "marital", "value_set": ['divorced', 'married','single']},
#         #     )
#         # )
#         # Create the expectation

#     # numerical features
#     if feature_group == 'market_numerical_features':

#         expectation_suite_bank.add_expectation(
#             ExpectationConfiguration(
#                 expectation_type="expect_table_column_count_to_equal",
#                 kwargs={"value": 49},
#             )
#         )
        
#         expected_column_names = market_columns_list_()[2:]

#         for col in expected_column_names:
#             expectation_suite_bank.add_expectation(
#                 ExpectationConfiguration(
#                     expectation_type="expect_column_values_to_be_of_type",
#                     kwargs={"column": col, "type_": "float64"},
#                 )
#             )

#         # target
#         if False:
#             # if feature_group == 'target':
                
#             #     expectation_suite_bank.add_expectation(
#             #         ExpectationConfiguration(
#             #             expectation_type="expect_column_distinct_values_to_be_in_set",
#             #             kwargs={"column": "y", "value_set": ['yes', 'no']},
#             #         )
#             #     ) 
#             pass
        
    

#     return expectation_suite_bank


# def to_feature_store(
#     data: pd.DataFrame,
#     group_name: str,
#     feature_group_version: int,
#     description: str,
#     group_description: dict,
#     validation_expectation_suite: ExpectationSuite,
#     credentials_input: dict
# ):
#     """
#     This function takes in a pandas DataFrame and a validation expectation suite,
#     performs validation on the data using the suite, and then saves the data to a
#     feature store in the feature store.

#     Args:
#         data (pd.DataFrame): Dataframe with the data to be stored
#         group_name (str): Name of the feature group.
#         feature_group_version (int): Version of the feature group.
#         description (str): Description for the feature group.
#         group_description (dict): Description of each feature of the feature group. 
#         validation_expectation_suite (ExpectationSuite): group of expectations to check data.
#         SETTINGS (dict): Dictionary with the settings definitions to connect to the project.
        
#     Returns:
#         A dictionary with the feature view version, feature view name and training dataset feature version.
#     """

#     # Connect to feature store.
#     project = hopsworks.login(
#         api_key_value=credentials_input["FS_API_KEY"], project=credentials_input["FS_PROJECT_NAME"]
#     )
#     feature_store = project.get_feature_store()

#     # Create feature group.
#     object_feature_group = feature_store.get_or_create_feature_group(
#         name=group_name,
#         version=feature_group_version,
#         description= description,
#         primary_key=["index"],
#         event_time="month_year",
#         online_enabled=False,
#         expectation_suite=validation_expectation_suite,
#     )
#     # Upload data.
#     object_feature_group.insert(
#         features=data,
#         overwrite=False,
#         write_options={
#             "wait_for_job": True,
#         },
#     )

#     # Add feature descriptions.

#     for description in group_description:
#         object_feature_group.update_feature_description(
#             description["name"], description["description"]
#         )

#     # Update statistics.
#     object_feature_group.statistics_config = {
#         "enabled": True,
#         "histograms": True,
#         "correlations": True,
#     }
#     object_feature_group.update_statistics_config()
#     object_feature_group.compute_statistics()

#     return object_feature_group


def preprocess_sales(
        data: pd.DataFrame, 
        parameters: Dict[str, Any],
        dummy_value) -> pd.DataFrame:
    
    logger = logging.getLogger(__name__)

    # Copy the DataFrame
    sales_copy = data.copy()

    # # This step already done while ingesting
    # # Convert 'Full_Date' column to datetime
    # sales_copy['Full_Date'] = pd.to_datetime(sales_copy['Full_Date'], format='%d-%m-%Y')

    # Group by both 'Full_Date' (month) and 'GCK' (product), and sum the sales
    sales_copy = sales_copy.groupby([sales_copy['full_date'].dt.to_period('M'), 'gck']).sum().reset_index()
    
    # # We will try to skip doing this on pipeline, because data types should be preserved by kedro
    # # Convert 'Full_Date' column to string
    # sales_copy['Full_Date'] = pd.to_datetime(sales_copy['Full_Date'].astype(str))
    #
    
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

    # Printing something from dataframe (usually columns)
    # dummy_value is for checking pipelines sequence
    pipeline_name = "preprocess_sales"
    f_verbose = True
    debug_on_success_(sales_copy, dummy_value, pipeline_name, f_verbose)

    return sales_copy, dummy_value


def preprocess_markets(
        data: pd.DataFrame, 
        parameters: Dict[str, Any],
        dummy_value) -> pd.DataFrame:
    
    logger = logging.getLogger(__name__)

    # Copy
    market_copy = data.copy()

    # # This is already done
    # # Function to parse date-like strings
    # def parse_date(date_str):
    #     month_str, year_str = date_str.split()
    #     month_map = {
    #         'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    #         'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    #     }
    #     month = month_map[month_str]
    #     year = int(year_str)
    #     return datetime(year, month, 1)

    # # Apply the function to the 'date' column
    # market_copy['Month Year'] = market_copy['Month Year'].apply(parse_date)

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

    # Printing something from dataframe (usually columns)
    # dummy_value is for checking pipelines sequence
    pipeline_name = "preprocess_markets"
    f_verbose = True
    debug_on_success_(market_copy, dummy_value, pipeline_name, f_verbose)

    return market_copy, dummy_value


def market_merge_german_gdp(
        market_data: pd.DataFrame,
        gdp_data: pd.DataFrame, 
        parameters: Dict[str, Any],
        dummy_value) -> pd.DataFrame:
    
    logger = logging.getLogger(__name__)

    # Copy
    market_copy = market_data.copy()
    gdp_copy = gdp_data.copy()

    # Convert the 'DATE' column to datetime format and set it as the index
    gdp_copy.set_index('month_year', inplace=True)

    # Set True/False whenever debug needed/or not
    if True:
        # Printing something from dataframe (usually columns)
        # dummy_value is for checking pipelines sequence
        pipeline_name = "market_merge_german_gdp"
        f_verbose = True
        debug_on_success_(market_copy, dummy_value, pipeline_name, f_verbose)

    # Set True/False whenever debug needed/or not
    if True:
        # Printing something from dataframe (usually columns)
        # dummy_value is for checking pipelines sequence
        pipeline_name = "market_merge_german_gdp"
        f_verbose = True
        debug_on_success_(gdp_copy, dummy_value, pipeline_name, f_verbose)

    #the .resample() method is applied to the index column of the DataFrame, which must be a datetime-like index
    # Resample the data to monthly frequency and forward fill missing values
    gdp_monthly = pd.DataFrame(gdp_copy.resample('MS').ffill()['gdp'] / 3)

    # Merge the datasets on the 'month_year' column
    market_copy = market_copy.merge(gdp_monthly.rename(columns={'gdp': 'german_gdp'}), on='month_year', how='left')

    logger.info(f"The MARKET merged with GERMAN GDP. Processed market contains {len(market_copy.columns)} columns.")

    pass

    # Printing something from dataframe (usually columns)
    # dummy_value is for checking pipelines sequence
    pipeline_name = "market_merge_german_gdp"
    f_verbose = True
    debug_on_success_(market_copy, dummy_value, pipeline_name, f_verbose)

    return market_copy, dummy_value
