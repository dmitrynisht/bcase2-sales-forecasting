from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_sales, preprocess_markets
# from .nodes import create_model_input_table, preprocess_companies, preprocess_shuttles, preprocess_ibm


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_sales,
                inputs=["sales_raw_data", "parameters"],
                outputs=["preprocessed_sales", "dummy"],
                name="preprocess_sales_node",
            ),
            node(
                func=preprocess_markets,
                inputs=["market_raw_data", "dummy", "parameters"],
                outputs="preprocessed_markets",
                name="preprocess_markets_node",
            ),

            # node(
            #     func=preprocess_shuttles,
            #     inputs="shuttles",
            #     outputs="preprocessed_shuttles",
            #     name="preprocess_shuttles_node",
            # ),
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
