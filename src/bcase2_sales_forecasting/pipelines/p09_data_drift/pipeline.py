from kedro.pipeline import Pipeline, node
from .nodes import load_data, generate_drift, evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=load_data,
                inputs=["X_train_data", "y_train_data"],
                outputs=["X_train", "y_train"],
                name="load_data_node",
            ),
            node(
                func=generate_drift,
                inputs=dict(
                    X="X_train",
                    y="y_train",
                    drift_factor="params:drift_factor"
                ),
                outputs=["drifted_X_train", "drifted_y_train"],
                name="generate_drift_node",
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "X_train",
                    "y_train",
                    "drifted_X_train",
                    "drifted_y_train",
                    "champion_model"
                ],
                outputs="metrics",
                name="evaluate_model_node",
            ),
        ]
    )
