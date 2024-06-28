"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_predict


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_predict,
                inputs=["production_model", 
                        "X_train_data", "X_val_data", "y_train_data", "y_val_data", "ingested_test_data",
                        "parameters"],
                outputs=["production_predictions", "production_rmse", "production_evalution_plot"],
                name="model_predict",
            ),
        ]
    )