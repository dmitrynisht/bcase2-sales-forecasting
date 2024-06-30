from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_sales, preprocess_markets, market_merge_german_gdp


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_sales,
                inputs=["ingested_sales", "parameters"],
                outputs="preprocessed_sales",
                name="preprocess_sales_node",
            ),
            node(
                func=preprocess_markets,
                inputs=["ingested_markets", "parameters"],
                outputs="preprocessed_markets",
                name="preprocess_markets_node",
            ),
            node(
                func=market_merge_german_gdp,
                inputs=["preprocessed_markets", "ingested_german_gdp", "parameters"],
                outputs="processed_markets",
                name="processed_markets_node",
            ),
        ]
    )
