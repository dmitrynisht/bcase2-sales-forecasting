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

    
def build_expectation_suite(expectation_suite_name: str, feature_group: str) -> ExpectationSuite:
    """
    Builder used to retrieve an instance of the validation expectation suite.
    
    Args:
        expectation_suite_name (str): A dictionary with the feature group name and the respective version.
        feature_group (str): Feature group used to construct the expectations.
            
    Returns:
        ExpectationSuite: A dictionary containing all the expectations for this particular feature group.
    """
        
    expectation_suite_bank = ExpectationSuite(
        expectation_suite_name=expectation_suite_name
    )
        
    if feature_group == "market_total_features":

        expected_column_names = market_columns_list_()

        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                # expectation_type="expect_table_columns_to_match_ordered_list",
                # kwargs={"column_list": expected_column_names},
                
                expectation_type="expect_column_values_to_be_unique",
                kwargs={
                        "column_map": {
                            "numeric_columns": expected_column_names[0],
                            "categorical_columns": expected_column_names[1:]
                        }
                }
            )
        )

        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_table_column_count_to_equal",
                kwargs={"value": 48},
            )
        )
        # # this intentionally left as example
        # expectation_suite_bank.add_expectation(
        #     ExpectationConfiguration(
        #         expectation_type="expect_column_distinct_values_to_be_in_set",
        #         kwargs={"column": "marital", "value_set": ['divorced', 'married','single']},
        #     )
        # )
        # Create the expectation

    #     # numerical features
    #     if feature_group == 'numerical_features':

    #         for i in ['age', 'duration','campaign', 'pdays', 'previous','balance']:
    #             expectation_suite_bank.add_expectation(
    #                 ExpectationConfiguration(
    #                     expectation_type="expect_column_values_to_be_of_type",
    #                     kwargs={"column": i, "type_": "int64"},
    #                 )
    #             )
    #         # age
    #         expectation_suite_bank.add_expectation(
    #                 ExpectationConfiguration(
    #                     expectation_type="expect_column_min_to_be_between",
    #                     kwargs={
    #                         "column": "age",
    #                         "min_value": 18,
    #                         "strict_min": False,
    #                     },
    #                 )
    #             )

    #     if feature_group == 'target':
            
    #         expectation_suite_bank.add_expectation(
    #             ExpectationConfiguration(
    #                 expectation_type="expect_column_distinct_values_to_be_in_set",
    #                 kwargs={"column": "y", "value_set": ['yes', 'no']},
    #             )
    #         ) 
        
    return expectation_suite_bank


def to_feature_store(
    data: pd.DataFrame,
    group_name: str,
    feature_group_version: int,
    description: str,
    group_description: dict,
    validation_expectation_suite: ExpectationSuite,
    credentials_input: dict
):
    """
    This function takes in a pandas DataFrame and a validation expectation suite,
    performs validation on the data using the suite, and then saves the data to a
    feature store in the feature store.

    Args:
        data (pd.DataFrame): Dataframe with the data to be stored
        group_name (str): Name of the feature group.
        feature_group_version (int): Version of the feature group.
        description (str): Description for the feature group.
        group_description (dict): Description of each feature of the feature group. 
        validation_expectation_suite (ExpectationSuite): group of expectations to check data.
        SETTINGS (dict): Dictionary with the settings definitions to connect to the project.
        
    Returns:
        A dictionary with the feature view version, feature view name and training dataset feature version.
    """

    # Connect to feature store.
    project = hopsworks.login(
        api_key_value=credentials_input["FS_API_KEY"], project=credentials_input["FS_PROJECT_NAME"]
    )
    feature_store = project.get_feature_store()

    # Create feature group.
    object_feature_group = feature_store.get_or_create_feature_group(
        name=group_name,
        version=feature_group_version,
        description= description,
        primary_key=["index"],
        event_time="Month Year",
        online_enabled=False,
        expectation_suite=validation_expectation_suite,
    )
    # Upload data.
    object_feature_group.insert(
        features=data,
        overwrite=False,
        write_options={
            "wait_for_job": True,
        },
    )

    # Add feature descriptions.

    for description in group_description:
        object_feature_group.update_feature_description(
            description["name"], description["description"]
        )

    # Update statistics.
    object_feature_group.statistics_config = {
        "enabled": True,
        "histograms": True,
        "correlations": True,
    }
    object_feature_group.update_statistics_config()
    object_feature_group.compute_statistics()

    return object_feature_group


def preprocess_sales(
    data: pd.DataFrame,
    parameters: Dict[str, Any]) -> pd.DataFrame:

    """Preprocesses the sales data.

    Args:
        data: Raw sales.
    Returns:
        Preprocessed sales
    """
    
    columns = data.columns.to_list()
    assert len(columns) == 3, "Wrong data collected"

    sales_data = data.copy()

    sales_data = sales_columns_naming_(sales_data)

    # Format data
    sales_data = full_date_col_(sales_data)

    # Format sales
    sales_data = sales_col_(sales_data)

    # Printing something from dataframe (usually columns)
    dummy_value = [0]
    debug_on_success_(sales_data, dummy_value)

    return sales_data, dummy_value


def preprocess_markets(
        data: pd.DataFrame,
        dummy_value, 
        parameters: Dict[str, Any], ) -> pd.DataFrame:
    
    """Preprocesses the market data.

    Args:
        data: Raw markets.
    Returns:
        Preprocessed market data
    """

    columns = data.columns.to_list()
    assert len(columns) == 48, "Wrong data collected"

    market_data = data.copy()

    market_data = market_columns_naming_(market_data)

    market_data = market_data.drop(market_data.index[0]).reset_index(drop=True)

    # We believed it would be easier to understand the columns with clearer names
    market_data['Month Year'] = market_data['Month Year'].str.strip()

    # Splitting the column into year and month
    market_data[['year', 'month']] = market_data['Month Year'].str.split('m', expand=True)

    # Converting year and month to datetime
    market_data['Month Year'] = pd.to_datetime(market_data['year'] + '-' + market_data['month'], format='%Y-%m')

    # # for feature store its better not to
    # # Formatting datetime to Month Year
    # market_data['Month Year'] = market_data['Month Year'].dt.strftime('%b %Y')

    # Dropping intermediate columns if needed
    market_data.drop(columns=['year', 'month'], inplace=True)

    # Define the list of columns and convert to float
    market_data.iloc[:, 1:] = market_data.iloc[:, 1:].astype(float)

    # print(market_data.columns)
    # print(60*"#")
    # print(len(market_data.columns))
    # print(len(market_columns_list_()))
    # print(60*"#")
    # print([x for x in market_columns_list_() if x not in market_data.columns])
    print(f'{30*"#"} {"preprocess_markets".upper} {30*"#"}')
    # Reset the index to convert the default index to a column
    market_data = market_data.reset_index()
    # Optionally, rename the new column if needed
    # market_data.rename(columns={'index': 'index'}, inplace=True)
    # print(market_data.index.name)
    # market_data = market_data.rename_axis('index')
    print(60*"#")

    logger.info(f"The dataset contains {len(market_data.columns)} columns.")
    
    validation_expectation_suite_market_total = build_expectation_suite("market_total_expectations","market_total_features")

    market_total_features_descriptions =[]

    if parameters["to_feature_store"]:

        object_fs_categorical_features = to_feature_store(
            market_data, "market_total_features",
            1, "Categorical Features",
            market_total_features_descriptions,
            validation_expectation_suite_market_total,
            credentials["feature_store"]
        )
        
    # Printing something from dataframe (usually columns)
    # dummy_value is for checking pipelines sequence
    # columns = market_data.columns.to_list()
    # print(len(columns), columns[:5])
    # print(market_data.iloc[:4, 0])
    debug_on_success_(market_data, dummy_value)

    return market_data

