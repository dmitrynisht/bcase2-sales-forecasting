from kedro.pipeline import Pipeline, node, pipeline

from .nodes import market_features_selection, compute_sales_lag_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=market_features_selection,
                inputs=["processed_markets", "parameters"],
                outputs="markets_best_features",
                name="markets_collinearity_elimination",
            ),
            node(
                func=compute_sales_lag_features,
                inputs=['preprocessed_sales', 'markets_best_features', 'parameters'],
                outputs=["features_with_sales_lag", "feature_importance_plot", "feature_correlation_plot"],
                name="compute_sales_lag_features",
            )
        ]
    )
