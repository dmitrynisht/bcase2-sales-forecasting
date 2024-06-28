from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_sales, preprocess_markets, market_merge_german_gdp


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_sales,
                inputs=["ingested_sales", "parameters", "03_dummy"],
                outputs=["preprocessed_sales", "04_dummy"],
                name="preprocess_sales_node",
            ),
            # node(
            #     func=preprocess_markets,
            #     inputs=["ingested_markets", "parameters", "04_dummy"],
            #     outputs=["preprocessed_markets", "05_dummy"],
            #     name="preprocess_markets_node",
            # ),
            # node(
            #     func=market_merge_german_gdp,
            #     inputs=["preprocessed_markets", "ingested_german_gdp", "parameters", "05_dummy"],
            #     outputs=["processed_markets", "06_dummy"],
            #     name="processed_markets_node",
            # ),
        ]
    )
