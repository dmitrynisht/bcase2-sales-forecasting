from kedro.pipeline import Pipeline, node, pipeline

from .nodes import ingest_sales, ingest_markets, preprocess_sales, preprocess_markets
# from .nodes import create_model_input_table, preprocess_companies, preprocess_shuttles, preprocess_ibm


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
                func=preprocess_sales,
                inputs=["ingested_sales", "parameters", "02_dummy"],
                outputs=["preprocessed_sales", "03_dummy"],
                name="preprocess_sales_node",
            ),
            node(
                func=preprocess_markets,
                inputs=["ingested_markets", "parameters", "03_dummy"],
                outputs=["preprocessed_markets", "04_dummy"],
                name="preprocess_markets_node",
            ),
            # node(
            #     func=create_model_input_table,
            #     inputs=["preprocessed_shuttles", "preprocessed_companies", "reviews"],
            #     outputs="model_input_table",
            #     name="create_model_input_table_node",
            # ),
            # node(
            #     func=preprocess_ibm,
            #     inputs="ibm_raw",
            #     outputs="ibm_transformed",
            #     name="preprocess_ibm_node",
            # ),
        ]
    )
