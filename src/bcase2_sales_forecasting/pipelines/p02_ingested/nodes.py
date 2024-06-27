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

    # numerical features
    if feature_group == 'sales_numerical_features':
        
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_table_column_count_to_equal",
                kwargs={"value": 3},
            )
        )

        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={"column": "sales_eur", "type_": "float64"},
            )
        )

        pass

    #
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

        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_table_column_count_to_equal",
                kwargs={"value": 49},
            )
        )
        
        expected_column_names = market_columns_list_()[2:]

        for col in expected_column_names:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": col, "type_": "float64"},
                )
            )

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
        
    # numerical features
    if feature_group == 'gdp_numerical_features':
        
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_table_column_count_to_equal",
                kwargs={"value": 2},
            )
        )

        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={"column": "gdp", "type_": "float64"},
            )
        )

        pass

    return expectation_suite_bank


def to_feature_store(
    data: pd.DataFrame,
    group_name: str,
    feature_group_version: int,
    description: str,
    primary_key: list,
    event_time: str,
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
        description=description,
        primary_key=primary_key,
        event_time=event_time,
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

    logger = logging.getLogger(__name__)

    pipeline_name = "ingest_sales"
    logger.info(f"{pipeline_name}")

    sales_copy = data.copy()

    sales_copy = sales_columns_naming_(sales_copy)

    # Format data
    sales_copy = full_date_col_(sales_copy)

    # Format sales
    sales_copy = sales_col_(sales_copy)

    sales_copy = columns_sanitation_(sales_copy)

    logger.info(f"The SALES dataset contains {len(sales_copy.columns)} columns.")

    if parameters["to_feature_store"]:

        categorical_dtypes = ['object','string','category']
        sales_numerical_features = sales_copy.select_dtypes(exclude=categorical_dtypes+['datetime']).columns.tolist()
        # sales_categorical_features = sales_copy.select_dtypes(include=categorical_dtypes).columns.tolist()

        # Reset the index to convert the default index to a column
        sales_copy = sales_copy.reset_index()

        validation_expectation_suite_sales_numerical = build_expectation_suite("sales_numerical_expectations","sales_numerical_features")
        # validation_expectation_suite_sales_categorical = build_expectation_suite("sales_categorical_expectations","sales_categorical_features")

        sales_numerical_feature_descriptions = {}
        # sales_categorical_feature_descriptions = {}

        primary_key = ["index"]
        event_time = "full_date"

        df_full_numeric = sales_copy[["index","full_date"] + sales_numerical_features]
        # df_full_categorical = sales_copy[["index","month_year"] + sales_categorical_features]

        object_fs_numerical_features = to_feature_store(
            df_full_numeric, "sales_numerical_features",
            1, "Numerical Features",
            primary_key,
            event_time,
            sales_numerical_feature_descriptions,
            validation_expectation_suite_sales_numerical,
            credentials["feature_store"]
        )

        logger.info(f"The SALES data delivered to feature store.")

    else:
        # actually could be removed from here because we have similar expectation
        #   but not identical
        columns = sales_copy.columns.to_list()
        assert len(columns) == 3, "Wrong data collected"

        # Reset the index to convert the default index to a column
        # sales_copy = sales_copy.reset_index()
        logger.info(f"{sales_copy.head(20)}")
        logger.info(f"{'#'*30 + ' sales_copy '.upper() + '#'*30}")

    dummy_value = [0]
    if parameters["debug_output"][pipeline_name]:
        # Printing something from dataframe (usually columns)
        # dummy_value is for checking pipelines sequence
        f_verbose = True
        debug_on_success_(sales_copy, dummy_value, pipeline_name, f_verbose)

    return sales_copy, dummy_value


def ingest_markets(
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        dummy_value) -> pd.DataFrame:
    
    """Ingest the market data.

    Args:
        data: Raw markets.
    Returns:
        Ingested market data
    """

    logger = logging.getLogger(__name__)

    pipeline_name = "ingest_markets"
    logger.info(f"{pipeline_name}")

    market_copy = data.copy()

    market_copy = market_columns_naming_(market_copy)

    market_copy = market_copy.drop(market_copy.index[0]).reset_index(drop=True)

    # We believed it would be easier to understand the columns with clearer names
    market_copy['Month Year'] = market_copy['Month Year'].str.strip()

    # Splitting the column into year and month
    market_copy[['year', 'month']] = market_copy['Month Year'].str.split('m', expand=True)

    # Converting year and month to datetime
    market_copy['Month Year'] = pd.to_datetime(market_copy['year'] + '-' + market_copy['month'], format='%Y-%m')

    # # for feature store its better not to do this
    # # Formatting datetime to Month Year
    # market_copy['Month Year'] = market_copy['Month Year'].dt.strftime('%b %Y')

    # Dropping intermediate columns if needed
    market_copy.drop(columns=['year', 'month'], inplace=True)

    market_copy = columns_sanitation_(market_copy)

    # Define the list of columns and convert to float
    # market_copy.iloc[:, 1:] = market_copy.iloc[:, 1:].astype(float) # this one generates depricatioin warning
    market_copy[market_copy.columns[1:]] = market_copy[market_copy.columns[1:]].apply(lambda x: x.astype(float))

    logger.info(f"The MARKET dataset contains {len(market_copy.columns)} columns.")
    
    if parameters["to_feature_store"]:

        categorical_dtypes = ['object','string','category']
        market_numerical_features = market_copy.select_dtypes(exclude=categorical_dtypes+['datetime']).columns.tolist()
        market_categorical_features = market_copy.select_dtypes(include=categorical_dtypes).columns.tolist()

        # Reset the index to convert the default index to a column
        market_copy = market_copy.reset_index()

        validation_expectation_suite_market_total = build_expectation_suite("market_total_expectations","market_total_features")
        validation_expectation_suite_market_numerical = build_expectation_suite("market_numerical_expectations","market_numerical_features")
        validation_expectation_suite_market_categorical = build_expectation_suite("market_categorical_expectations","market_categorical_features")

        # market_total_features_descriptions = {}
        market_numerical_feature_descriptions = {}
        # market_categorical_feature_descriptions = {}
        # target_feature_descriptions = {}

        primary_key = ["index"]
        event_time = "month_year"

        df_full_numeric = market_copy[["index","month_year"] + market_numerical_features]
        # df_full_categorical = market_copy[["index","month_year"] + market_categorical_features]
        # df_full_target = df_full[["index","month_year"] + [parameters["target_column"]]]

        object_fs_numerical_features = to_feature_store(
            df_full_numeric, "market_numerical_features",
            1, "Numerical Features",
            primary_key,
            event_time,
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

        logger.info(f"The MARKET data delivered to feature store.")

    else:
        # actually could be removed from here because we have similar expectation
        #   but not identical
        columns = data.columns.to_list()
        assert len(columns) == 48, "Wrong data collected"
        
        # Reset the index to convert the default index to a column
        market_copy = market_copy.reset_index()

    # Set True/False whenever debug needed/or not
    if parameters["debug_output"][pipeline_name]:
        # Printing something from dataframe (usually columns)
        # dummy_value is for checking pipelines sequence
        f_verbose = True
        debug_on_success_(market_copy, dummy_value, pipeline_name, f_verbose)

    return market_copy, dummy_value


def ingest_gdp(
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        dummy_value) -> pd.DataFrame:
    
    """Ingest the German_GDP data.

    Args:
        data: Raw German_GDP.
    Returns:
        Ingested German_GDP data
    """

    logger = logging.getLogger(__name__)

    pipeline_name = "ingest_german_gdp"
    logger.info(f"{pipeline_name}")

    gdp_copy = data.copy()

    # Convert the 'DATE' column to datetime format and set it as the index
    gdp_copy['Month Year'] = pd.to_datetime(gdp_copy['Date'])
    gdp_copy.drop("Date", axis=1, inplace=True)
    
    gdp_copy = columns_sanitation_(gdp_copy)

    logger.info(f"The GERMAN GDP dataset contains {len(gdp_copy.columns)} columns.")

    if parameters["to_feature_store"]:

        categorical_dtypes = ['object','string','category']
        gdp_numerical_features = gdp_copy.select_dtypes(exclude=categorical_dtypes+['datetime']).columns.tolist()
        # market_categorical_features = market_copy.select_dtypes(include=categorical_dtypes).columns.tolist()

        # Reset the index to convert the default index to a column
        # gdp_copy.set_index('month_year', inplace=True)# check if it will work to have one column both index and datetime
        gdp_copy = gdp_copy.reset_index()

        # validation_expectation_suite_market_total = build_expectation_suite("market_total_expectations","market_total_features")
        validation_expectation_suite_gdp_numerical = build_expectation_suite("gdp_numerical_expectations","gdp_numerical_features")
        # validation_expectation_suite_gdp_categorical = build_expectation_suite("gdp_categorical_expectations","gdp_categorical_features")

        # market_total_features_descriptions = {}
        gdp_numerical_feature_descriptions = {}
        # market_categorical_feature_descriptions = {}
        # # target_feature_descriptions = {}

        primary_key = ["index"]
        event_time = "month_year"

        df_full_numeric = gdp_copy[["index","month_year"] + gdp_numerical_features]
        # df_full_categorical = market_copy[["index","month_year"] + market_categorical_features]
        # # df_full_target = df_full[["index","month_year"] + [parameters["target_column"]]]

        object_fs_numerical_features = to_feature_store(
            df_full_numeric, "gdp_numerical_features",
            1, "Numerical Features",
            primary_key,
            event_time,
            gdp_numerical_feature_descriptions,
            validation_expectation_suite_gdp_numerical,
            credentials["feature_store"]
        )

        logger.info(f"The GERMAN GDP data delivered to feature store.")

    else:
        # actually could be removed from here because we have similar expectation
        #   but not identical
        columns = gdp_copy.columns.to_list()
        assert len(columns) == 2, "Wrong data collected"
        
        # Reset the index to convert the default index to a column
        gdp_copy = gdp_copy.reset_index()

    # Set True/False whenever debug needed/or not
    if parameters["debug_output"][pipeline_name]:
        # Printing something from dataframe (usually columns)
        # dummy_value is for checking pipelines sequence
        f_verbose = True
        debug_on_success_(gdp_copy, dummy_value, pipeline_name, f_verbose)

    return gdp_copy, dummy_value
