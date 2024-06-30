"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  split_into_training_validation_data, prepare_full_training_data, prepare_test_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= split_into_training_validation_data,
                inputs=["features_with_sales_lag", "parameters"],
                outputs= ["df_train", "df_val"],
                name="split_into_training_validation_data",
            ),
            node(
                func= prepare_full_training_data,
                inputs=["features_with_sales_lag", "parameters"],
                outputs= "df_train_full",
                name="prepare_full_training_data",
            ),
            node(
                func= prepare_test_data,
                inputs=["ingested_test_data", "parameters"],
                outputs= "df_test",
                name="prepare_test_data",
            ),
        ]
    )
