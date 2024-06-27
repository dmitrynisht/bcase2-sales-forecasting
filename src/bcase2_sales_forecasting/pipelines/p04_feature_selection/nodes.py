from typing import Any, Dict, Tuple
import logging
from pathlib import Path
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
import pandas as pd
import numpy as np
from .utils import *

conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]


logger = logging.getLogger(__name__)


def sales_features_selection(
        sales_data: pd.DataFrame,
        parameters: Dict[str, Any],
        dummy_value) -> pd.DataFrame:
    
    logger = logging.getLogger(__name__)

    pipeline_name = "sales_features_selection"

    logger.info(f"{pipeline_name}")

    # Copy
    sales_copy = sales_data.copy()

    pass

    # Set True/False whenever debug needed/or not
    if parameters["debug_output"][pipeline_name]:
        # Printing something from dataframe (usually columns)
        # dummy_value is for checking pipelines sequence
        f_verbose = True
        debug_on_success_(sales_copy, dummy_value, pipeline_name, f_verbose)

    return sales_copy


def market_features_selection(
        market_data: pd.DataFrame,
        parameters: Dict[str, Any],
        dummy_value) -> pd.DataFrame:
    
    logger = logging.getLogger(__name__)

    pipeline_name = "market_features_selection"

    logger.info(f"{pipeline_name}")

    # Copy
    market_copy = market_data.copy()

    # Removing correlated features

    # Select only numerical features
    numerical_features = market_copy.select_dtypes(include=[np.number])

    # Calculate correlation matrix
    correlation_matrix = numerical_features.corr().abs()

    # Create a mask to select the upper triangle of correlation matrix
    mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)

    # Select upper triangle of correlation matrix using the mask
    upper_triangle = correlation_matrix.where(mask)

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
    logger.info(f"{pipeline_name.upper()}. features to drop:")
    logger.info(f"{to_drop}")

    # Drop highly correlated features
    market_copy = market_copy.drop(columns=to_drop)
    
    logger.info(f"The {'market dataset'.upper()} processing finished.")

    pass

    # Set True/False whenever debug needed/or not
    if parameters["debug_output"][pipeline_name]:
        # Printing something from dataframe (usually columns)
        # dummy_value is for checking pipelines sequence
        f_verbose = True
        debug_on_success_(market_copy, dummy_value, pipeline_name, f_verbose)

    return market_copy, dummy_value
