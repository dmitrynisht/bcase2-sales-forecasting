from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_sales, preprocess_markets, market_merge_german_gdp
# from .nodes import ingest_sales, ingest_markets
# from .nodes import create_model_input_table, preprocess_companies, preprocess_shuttles, preprocess_ibm


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_sales,
                inputs=["ingested_sales", "parameters", "03_dummy"],
                outputs=["preprocessed_sales", "04_dummy"],
                name="preprocess_sales_node",
            ),
            node(
                func=preprocess_markets,
                inputs=["ingested_markets", "parameters", "04_dummy"],
                outputs=["preprocessed_markets", "05_dummy"],
                name="preprocess_markets_node",
            ),
            node(
                func=market_merge_german_gdp,
                inputs=["preprocessed_markets", "ingested_german_gdp", "parameters", "05_dummy"],
                outputs=["processed_markets", "06_dummy"],
                name="processed_markets_node",
            ),
            # node(
            #     func=preprocess_ibm,
            #     inputs="ibm_raw",
            #     outputs="ibm_transformed",
            #     name="preprocess_ibm_node",
            # ),
        ]
    )
