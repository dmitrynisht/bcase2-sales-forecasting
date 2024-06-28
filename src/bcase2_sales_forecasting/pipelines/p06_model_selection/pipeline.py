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
                inputs=["X_train_data","X_val_data","y_train_data","y_val_data",
                        "parameters"],
                outputs=["champion_model", "champion_model_parameters"],
                name="model_selection",
            ),
        ]
    )