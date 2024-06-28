
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

logger = logging.getLogger(__name__)

def model_train(champion_model,
                champion_model_parameters: Dict[str, Any],
                X_train: pd.DataFrame, 
                X_val: pd.DataFrame, 
                y_train: pd.DataFrame, 
                y_val: pd.DataFrame,
                parameters: Dict[str, Any]):
    """Trains a model on the given data and saves it to the given model path.

    Args:
    --
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): Test features.
        y_train (pd.DataFrame): Training target.
        y_val (pd.DataFrame): Test target.

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

    results_dict = {}
    with mlflow.start_run(experiment_id=experiment_id, nested=True):

        class_name = champion_model.__class__.__name__

        # Reinitialize the model
        if class_name == 'NeuralProphet':
            model = NeuralProphet()
        elif class_name == 'Prophet':
            model = Prophet()
        else:
            raise ValueError(f"Unsupported model class: {class_name}")

        X_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
        y_full = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

        df_full = pd.DataFrame({
                                'ds': X_full['Full_Date'],
                                'y': y_full[parameters['target_column']]
                                })

        fit_params = champion_model_parameters

        model.fit(df_full, **fit_params)

        # saving results in dict
        results_dict['model'] = class_name
        # results_dict['train_score'] = acc_train
        # results_dict['test_score'] = acc_test
        print('RESULTS: ', results_dict)

        # logging in mlflow
        run_id = mlflow.last_active_run().info.run_id
        logger.info(f"Logged train model in run {run_id}")


    return model, fit_params