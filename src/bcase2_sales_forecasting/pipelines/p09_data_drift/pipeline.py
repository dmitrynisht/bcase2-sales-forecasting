from kedro.pipeline import Pipeline, node
from .nodes import detect_drift

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=detect_drift,
                inputs=["features_with_sales_lag", "df_train_full", "parameters"],
                outputs="drift_detection_result",
                name="drift_detection_node",
            ),
        ]
    )