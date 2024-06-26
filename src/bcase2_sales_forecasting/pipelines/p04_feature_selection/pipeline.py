from kedro.pipeline import Pipeline, node, pipeline

from .nodes import market_features_selection


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=market_features_selection,
                inputs=["processed_markets", "parameters", "06_dummy"],
                outputs=["markets_feature_selection", "07_dummy"],
                name="markets_collinearity_elimination",
            ),
        ]
    )
