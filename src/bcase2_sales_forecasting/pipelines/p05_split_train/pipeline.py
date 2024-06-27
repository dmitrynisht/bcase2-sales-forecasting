"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= split_data,
                inputs=["dev_sales_p1","parameters"],
                outputs= ["X_train_data","X_val_data","y_train_data","y_val_data"],
                name="split_data",
            ),
        ]
    )
