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
    
    expectation_suite_bank.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_table_column_count_to_equal",
            kwargs={"value": 49},
        )
    )

    if feature_group == "market_total_features":

        expected_column_names = market_columns_list_()

        # # to improve, we don't need unique values within columns
        # expectation_suite_bank.add_expectation(
        #     ExpectationConfiguration(
        #         # expectation_type="expect_table_columns_to_match_ordered_list",
        #         # kwargs={"column_list": expected_column_names},
                
        #         expectation_type="expect_column_values_to_be_unique",
        #         kwargs={
        #                 "column_map": {
        #                     "numeric_columns": expected_column_names[2:],
        #                     # "categorical_columns": expected_column_names[1:]
        #                 }
        #         }
        #     )
        # )

        # expectation_suite_bank.add_expectation(
        #     ExpectationConfiguration(
        #         expectation_type="expect_table_column_count_to_equal",
        #         kwargs={"value": 49},
        #     )
        # )

        # # this intentionally left as example
        # expectation_suite_bank.add_expectation(
        #     ExpectationConfiguration(
        #         expectation_type="expect_column_distinct_values_to_be_in_set",
        #         kwargs={"column": "marital", "value_set": ['divorced', 'married','single']},
        #     )
        # )
        # Create the expectation

    # numerical features
    if feature_group == 'market_numerical_features':

        expected_column_names = market_columns_list_()[2:]

        for col in expected_column_names:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": col, "type_": "float64"},
                )
            )

        # # age
        # expectation_suite_bank.add_expectation(
        #         ExpectationConfiguration(
        #             expectation_type="expect_column_min_to_be_between",
        #             kwargs={
        #                 "column": "age",
        #                 "min_value": 18,
        #                 "strict_min": False,
        #             },
        #         )
        #     )

        # target
        if False:
            # if feature_group == 'target':
                
            #     expectation_suite_bank.add_expectation(
            #         ExpectationConfiguration(
            #             expectation_type="expect_column_distinct_values_to_be_in_set",
            #             kwargs={"column": "y", "value_set": ['yes', 'no']},
            #         )
            #     ) 
            pass
        
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
        event_time="month_year",
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


def ingest_sales(
    data: pd.DataFrame,
    parameters: Dict[str, Any]) -> pd.DataFrame:

    """Ingest the sales data.

    Args:
        data: Raw sales.
    Returns:
        Ingested sales
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


def ingest_markets(
        data: pd.DataFrame,
        dummy_value, 
        parameters: Dict[str, Any], ) -> pd.DataFrame:
    
    """Ingest the market data.

    Args:
        data: Raw markets.
    Returns:
        Ingested market data
    """

    # actually could be removed from here because we have similar expectation
    #   but not identical
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

    # # for feature store its better not to do this
    # # Formatting datetime to Month Year
    # market_data['Month Year'] = market_data['Month Year'].dt.strftime('%b %Y')

    # Dropping intermediate columns if needed
    market_data.drop(columns=['year', 'month'], inplace=True)

    market_data = market_columns_sanitation_(market_data)

    # Define the list of columns and convert to float
    market_data.iloc[:, 1:] = market_data.iloc[:, 1:].astype(float)

    # Set True/False whenever debug needed/or not
    if False:
        print_to_debug_(market_data, None)

    logger.info(f"The dataset contains {len(market_data.columns)} columns.")
    
    categorical_dtypes = ['object','string','category']
    market_numerical_features = market_data.select_dtypes(exclude=categorical_dtypes+['datetime']).columns.tolist()
    market_categorical_features = market_data.select_dtypes(include=categorical_dtypes).columns.tolist()

    # Reset the index to convert the default index to a column
    market_data = market_data.reset_index()

    validation_expectation_suite_market_total = build_expectation_suite("market_total_expectations","market_total_features")
    validation_expectation_suite_market_numerical = build_expectation_suite("market_numerical_expectations","market_numerical_features")
    validation_expectation_suite_market_categorical = build_expectation_suite("market_categorical_expectations","market_categorical_features")

    # market_total_features_descriptions =[]
    market_numerical_feature_descriptions =[]
    market_categorical_feature_descriptions =[]
    # target_feature_descriptions =[]

    df_full_numeric = market_data[["index","month_year"] + market_numerical_features]
    df_full_categorical = market_data[["index","month_year"] + market_categorical_features]
    # df_full_target = df_full[["index","month_year"] + [parameters["target_column"]]]

    if parameters["to_feature_store"]:
        
        object_fs_numerical_features = to_feature_store(
            df_full_numeric, "market_numerical_features",
            1, "Numerical Features",
            market_numerical_feature_descriptions,
            validation_expectation_suite_market_numerical,
            credentials["feature_store"]
        )

        # object_fs_categorical_features = to_feature_store(
        #     market_data, "market_total_features",
        #     1, "Categorical Features",
        #     market_total_features_descriptions,
        #     validation_expectation_suite_market_total,
        #     credentials["feature_store"]
        # )

    # Printing something from dataframe (usually columns)
    # dummy_value is for checking pipelines sequence
    debug_on_success_(market_data, dummy_value)

    return market_data


def preprocess_sales(
        data: pd.DataFrame,
        dummy_value, 
        parameters: Dict[str, Any], ) -> pd.DataFrame:
    
    pass

    return data, dummy_value