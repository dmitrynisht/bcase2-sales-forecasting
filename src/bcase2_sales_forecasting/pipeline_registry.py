"""Project pipelines."""
from typing import Dict

# from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from bcase2_sales_forecasting.pipelines import (
    data_preprocessing as preprocessing
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # pipelines = find_pipelines()
    # pipelines["__default__"] = sum(pipelines.values())
    # return pipelines
    
    data_preprocessing = preprocessing.create_pipeline()
    return {
  
        "data_preprocessing": data_preprocessing,

    }