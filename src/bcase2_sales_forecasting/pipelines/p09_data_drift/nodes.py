import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from typing import Dict, Tuple, Any
import logging

from alibi_detect.cd import KSDrift
import pandas as pd

logger = logging.getLogger(__name__)

def detect_drift(historical_data: pd.DataFrame,
                 new_data: pd.DataFrame,
                 parameters: Dict[str, Any],
                 ) -> dict:
    """
    Detect drift in a specific feature using Kolmogorov-Smirnov test.
    
    Args:
        historical_data (pd.DataFrame): The historical data (training data).
        new_data (pd.DataFrame): The new data to check for drift.
        parameters dict (str, Any): Parameters with feature to check and p_val_drift_threshold.
    
    Returns:
        dict: A dictionary with drift detection results.
    """


    # Extract the feature column from both datasets
    historical_feature = historical_data[parameters['historical_feature_name']].values
    new_feature = new_data[parameters['prepared_feature_name']].values
    
    # Initialize KSDrift detector
    cd = KSDrift(historical_feature, p_val=parameters['p_val_drift_threshold'])
    
    # Detect drift
    preds = cd.predict(new_feature)

    # Log warning if drift is detected
    if preds['data']['is_drift']:
        logger.warning('------------------------------------------------------------------------------')
        logger.warning(f"Data drift detected for feature '{parameters['historical_feature_name']}' with p-value {preds['data']['p_val'][0]}")
        logger.warning('------------------------------------------------------------------------------')
    else:
        logger.info(f"No detection of data drift!")

    # Convert the result to a DataFrame
    result = pd.DataFrame({
        'is_drift': [preds['data']['is_drift']],
        'p_value': [preds['data']['p_val']],
        'threshold': [preds['data']['threshold']]
    })
    
    # Return the result
    return result

