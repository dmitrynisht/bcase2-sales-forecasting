"""Project pipelines."""
from typing import Dict

# from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from bcase2_sales_forecasting.pipelines import (
    p01_raw_data_unit_tests as raw_data_tests,
    p02_ingested as ingested,
    p03_data_preprocessing as preprocessing
#
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # pipelines = find_pipelines()
    # pipelines["__default__"] = sum(pipelines.values())
    # return pipelines
    
    raw_data_unit_tests = raw_data_tests.create_pipeline()
    raw_data_ingested = ingested.create_pipeline()
    preprocess_data = preprocessing.create_pipeline()

    return {
        "raw_data_unit_tests": raw_data_unit_tests,
        "raw_data_ingested": raw_data_ingested,
        "preprocess_data": preprocess_data,
        
        # "long_pipe": raw_data_unit_tests + raw_data_ingested + preprocess_data,
    }