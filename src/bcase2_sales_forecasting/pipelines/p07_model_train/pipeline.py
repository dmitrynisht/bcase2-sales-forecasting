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
                        "X_train_data","X_val_data", "y_train_data","y_val_data",
                        "parameters"],
                outputs=["production_model", "production_model_parameters"],
                name="model_train",
            ),
        ]
    )