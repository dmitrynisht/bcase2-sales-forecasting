from kedro.pipeline import Pipeline, node, pipeline

from .nodes import market_features_selection, sales_features_selection, compute_sales_lag_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=market_features_selection,
                inputs=["processed_markets", "parameters", "06_dummy"],
                outputs=["markets_best_features", "07_dummy"],
                name="markets_collinearity_elimination",
            ),
            # node(
            #     func=sales_features_selection,
            #     inputs=["processed_sales", "parameters", "07_dummy"],
            #     outputs=["sales_best_features", "08_dummy"],
            #     name="markets_collinearity_elimination",
            # ),
            node(
                func=compute_sales_lag_features,
                inputs=['preprocessed_sales', 'markets_feature_selection', 'parameters'],
                outputs=["features_with_sales_lag"],
                name="compute_sales_lag_features",
            )
        ]
    )
