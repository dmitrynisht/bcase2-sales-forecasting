"""Project pipelines."""
from typing import Dict

# from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from bcase2_sales_forecasting.pipelines import (
    p01_raw_data_unit_tests as raw_data_tests_pipeline,
    p02_ingested as ingested_pipeline,
    p03_data_preprocessing as data_preprocessing_pipeline,
    p04_feature_selection as feature_selection_pipeline,
    p05_split_train as split_data_pipeline,
    p06_model_selection as model_selection_pipeline,
    p07_model_train as model_train_pipeline,
    p08_model_predict as model_predict_pipeline,
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
    
    raw_data_unit_tests = raw_data_tests_pipeline.create_pipeline()
    raw_data_ingested = ingested_pipeline.create_pipeline()
    preprocess_data = data_preprocessing_pipeline.create_pipeline()
    feature_selection = feature_selection_pipeline.create_pipeline()
    split_data = split_data_pipeline.create_pipeline()
    model_selection = model_selection_pipeline.create_pipeline()
    model_train = model_train_pipeline.create_pipeline()
    model_predict = model_predict_pipeline.create_pipeline()

    return {
        "raw_data_unit_tests": raw_data_unit_tests,
        "raw_data_ingested": raw_data_ingested,
        "preprocess_data": preprocess_data,
        "feature_selection": feature_selection,
        "split_data":  split_data,
        "model_selection": model_selection,
        "model_train": model_train,
        "model_predict": model_predict,
        
        "long_pipe": raw_data_unit_tests + raw_data_ingested + preprocess_data + feature_selection + split_data + model_selection + model_train +model_predict,
    }