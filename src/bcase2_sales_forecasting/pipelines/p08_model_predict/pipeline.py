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
                        "df_train_full", "df_test", "parameters"],
                outputs=["production_predictions", "production_rmse", "production_evalution_plot"],
                name="model_predict",
            ),
        ]
    )