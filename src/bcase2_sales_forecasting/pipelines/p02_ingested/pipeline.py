"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  ingest_sales, ingest_markets, ingest_gdp , ingest_test_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=ingest_sales,
                inputs=["sales_raw_data", "parameters"],
                outputs=["ingested_sales", "01_dummy"],
                name="ingest_sales_node",
            ),
            node(
                func=ingest_markets,
                inputs=["market_raw_data", "parameters", "01_dummy"],
                outputs=["ingested_markets", "02_dummy"],
                name="ingest_markets_node",
            ),
            node(
                func=ingest_gdp,
                inputs=["german_gdp_raw_data", "parameters", "02_dummy"],
                outputs=["ingested_german_gdp", "03_dummy"],
                name="ingest_german_gdp_node",
            ),
            node(
                func=ingest_test_data,
                inputs=["test_raw_data", "parameters"],
                outputs="ingested_test_data",
                name="ingest_test_node",
            ),
        ]
    )


