
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np
import pickle
import yaml
import warnings
warnings.filterwarnings("ignore", category=Warning)
import mlflow
import matplotlib.pyplot as plt

# Time series Models
from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed
from prophet import Prophet

from .utils import train_fb_prophet, train_neural_prophet

logger = logging.getLogger(__name__)

def model_train(champion_model,
                champion_model_parameters: Dict[str, Any],
                df_full: pd.DataFrame,
                parameters: Dict[str, Any]):
    """Trains a model on the given data and saves it to the given model path.

    Args:
    --  
        champion_model: champion model to train.
        champion_model_parameters (Dict): Fit parameter of champion model
        df_full (pd.DataFrame): Full training data.

    Returns:
    --
        model (pickle): Trained models.
        model parameters(json): Trained model parameters.
    """

    # enable autologging
    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.load(f, Loader=yaml.loader.SafeLoader)['tracking']['experiment']['name']
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        logger.info(experiment_id)

    logger.info('Starting first step of model model training of champion model.')

    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

    class_name = champion_model.__class__.__name__
    
    with mlflow.start_run(experiment_id=experiment_id, nested=True):

        # Reinitialize the model
        if class_name == 'NeuralProphet':
            set_random_seed(parameters['random_state'])
            model = NeuralProphet()
            model = train_neural_prophet(model, df_full, parameters)

        elif class_name == 'Prophet':
            model = Prophet()
            model = train_fb_prophet(model, df_full, parameters)
        else:
            raise ValueError(f"Unsupported model class: {class_name}")

        fit_params = champion_model_parameters

        # logging in mlflow
        run_id = mlflow.last_active_run().info.run_id
        logger.info(f"Logged train model in run {run_id}")

    return model, fit_params