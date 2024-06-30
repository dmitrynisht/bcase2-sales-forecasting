"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_train


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_train,
                inputs=["champion_model", "champion_model_parameters", 
                        "df_train_full", "parameters"],
                outputs=["production_model", "production_model_parameters"],
                name="model_train",
            ),
        ]
    )