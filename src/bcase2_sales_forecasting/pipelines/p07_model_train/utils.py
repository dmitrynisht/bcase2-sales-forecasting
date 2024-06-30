import pandas as pd
from typing import Dict, Any
import numpy as np  
import warnings
from neuralprophet import set_random_seed
import mlflow

warnings.filterwarnings("ignore", category=Warning)


def train_neural_prophet(model,
                    df_full: pd.DataFrame, 
                    parameters: Dict[str, Any]):
    """
    """
     # Define fit parameters
    fit_params = parameters['fit_params_NeuralProphet']

    # Log the fit parameters
    mlflow.log_params(fit_params)
    
    set_random_seed(parameters['random_state'])
    model.fit(df_full, freq=fit_params['freq'], batch_size=fit_params['batch_size'],
                metrics=fit_params['metrics'], progress=fit_params['progress'],
                epochs=fit_params['epochs'])

    return model

def train_fb_prophet(model,
                df_full: pd.DataFrame, 
                parameters: Dict[str, Any]):
    """
    """
    # Define fit parameters
    fit_params = parameters['fit_params_Prophet']

    # Log the fit parameters
    mlflow.log_params(fit_params)

    model.fit(df_full)

    return model