from kedro.pipeline import Pipeline, node, pipeline

from .nodes import market_features_selection, compute_sales_lag_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=market_features_selection,
                inputs=["processed_markets", "parameters", "06_dummy"],
                outputs=["markets_feature_selection", "07_dummy"],
                name="markets_collinearity_elimination",
            ),
            node(
                func=compute_sales_lag_features,
                inputs=['preprocessed_sales', 'markets_feature_selection', 'parameters'],
                outputs=["features_with_sales_lag"],
                name="compute_sales_lag_features",
            )
        ]
    )
