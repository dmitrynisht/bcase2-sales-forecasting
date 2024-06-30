"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_selection


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_selection,
                inputs=["df_train", "df_val", "parameters"],
                outputs=["champion_model", "champion_model_parameters"],
                name="model_selection",
            ),
        ]
    )