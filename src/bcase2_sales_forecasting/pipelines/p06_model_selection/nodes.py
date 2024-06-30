import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np  
import yaml
import pickle
import warnings
warnings.filterwarnings("ignore", category=Warning)

# Time series Models
from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed
from prophet import Prophet

import mlflow

from .utils import train_fb_prophet, train_neural_prophet

logger = logging.getLogger(__name__)

     
def model_selection(df_train: pd.DataFrame, 
                    df_val: pd.DataFrame,
                    parameters: Dict[str, Any]):
    """Trains a model on the given data and saves it to the given model path.

    Args:
    --
        df_train (pd.DataFrame): Training data.
        df_val (pd.DataFrame): Validation data.
        parameters (dict): Parameters defined in parameters.yml.

    Returns:
    --
        models (dict): Dictionary of trained models.
        scores (pd.DataFrame): Dataframe of model scores.
    """
   
    models_dict = {
        'NeuralProphet': NeuralProphet(),
        'FB_Prophet': Prophet(),
    }

    initial_results = {}   
    model_parameters = {} 

    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.load(f, Loader=yaml.loader.SafeLoader)['tracking']['experiment']['name']
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        logger.info(experiment_id)


    logger.info('Starting first step of model selection : Comparing between model types')

    for model_name, model in models_dict.items():

        with mlflow.start_run(experiment_id=experiment_id, nested=True):
            mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)


            if model_name == 'NeuralProphet':
                model, model_rmse_val_last_epoch, fit_params = train_neural_prophet(model=model,
                                                                                    df_train=df_train,
                                                                                    df_val=df_val,
                                                                                    parameters=parameters)

            elif model_name == 'FB_Prophet':
                model, model_rmse_val_last_epoch, fit_params = train_fb_prophet(model=model,
                                                                                df_train=df_train,
                                                                                df_val=df_val,
                                                                                parameters=parameters)
            else:
                logger.warning(f"Implementation missing for {model_name}.")

            # save results of each model
            initial_results[model_name] = model_rmse_val_last_epoch
            model_parameters[model_name] = fit_params

            run_id = mlflow.last_active_run().info.run_id
            logger.info(f"Logged model : {model_name} in run {run_id}")
    
    
    print('BEST_RESULTS: ', initial_results)
    best_model_name = min(initial_results, key=initial_results.get)
    best_model = models_dict[best_model_name]
    best_model_parameters = model_parameters[best_model_name] 

    logger.info(f"Best model is {best_model_name} with score {initial_results[best_model_name]}")

    return best_model, best_model_parameters

    # logger.info('Starting second step of model selection : Hyperparameter tuning')

    # # Perform hyperparameter tuning with GridSearchCV
    # param_grid = parameters['hyperparameters'][best_model_name]
    # with mlflow.start_run(experiment_id=experiment_id,nested=True):
    #     gridsearch = GridSearchCV(best_model, param_grid, cv=2, scoring='accuracy', n_jobs=-1)
    #     gridsearch.fit(X_train, y_train)
    #     best_model = gridsearch.best_estimator_


    # logger.info(f"Hypertunned model score: {gridsearch.best_score_}")
    # pred_score = accuracy_score(y_test, best_model.predict(X_test))

    # if champion_dict['test_score'] < pred_score:
    #     logger.info(f"New champion model is {best_model_name} with score: {pred_score} vs {champion_dict['test_score']} ")
    #     return best_model
    # else:
    #     logger.info(f"Champion model is still {champion_dict['regressor']} with score: {champion_dict['test_score']} vs {pred_score} ")
    #     return champion_model

