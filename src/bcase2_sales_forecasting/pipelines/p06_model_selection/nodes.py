
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

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import mlflow

logger = logging.getLogger(__name__)

     
def model_selection(X_train: pd.DataFrame, 
                    X_val: pd.DataFrame, 
                    y_train: pd.DataFrame, 
                    y_val: pd.DataFrame,
                    parameters: Dict[str, Any]):
    
    
    """Trains a model on the given data and saves it to the given model path.

    Args:
    --
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): Validation features.
        y_train (pd.DataFrame): Training target.
        y_val (pd.DataFrame): Validation target.
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

    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.load(f, Loader=yaml.loader.SafeLoader)['tracking']['experiment']['name']
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        logger.info(experiment_id)


    logger.info('Starting first step of model selection : Comparing between model types')

    for model_name, model in models_dict.items():

        with mlflow.start_run(experiment_id=experiment_id, nested=True):
            mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

            if model_name == 'NeuralProphet':
                model, model_metrics, model_rmse_val_last_epoch = train_neural_prophet(model=model,
                                                                                       X_train=X_train,
                                                                                       X_val=X_val,
                                                                                       y_train=y_train,
                                                                                       y_val=y_val,
                                                                                       parameters=parameters)

            elif model_name == 'FB_Prophet':
                model, model_metrics, model_rmse_val_last_epoch = train_fb_prophet(model=model,
                                                                                       X_train=X_train,
                                                                                       X_val=X_val,
                                                                                       y_train=y_train,
                                                                                       y_val=y_val,
                                                                                       parameters=parameters)
            else:
                logger.warning(f"Implementation missing for {model_name}.")

            # save results of each model
            initial_results[model_name] = model_rmse_val_last_epoch

            run_id = mlflow.last_active_run().info.run_id
            logger.info(f"Logged model : {model_name} in run {run_id}")
    
    
    print('BEST_RESULTS: ', initial_results)
    best_model_name = min(initial_results, key=initial_results.get)
    best_model = models_dict[best_model_name]

    logger.info(f"Best model is {best_model_name} with score {initial_results[best_model_name]}")

    return best_model

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

def train_neural_prophet(model,
                    X_train: pd.DataFrame, 
                    X_val: pd.DataFrame, 
                    y_train: pd.DataFrame, 
                    y_val: pd.DataFrame,
                    parameters: Dict[str, Any]):
    """
    """
    df_train = pd.DataFrame({
                            'ds': X_train['Full_Date'],
                            'y': y_train[parameters['target_column']]
                            })

    df_val = pd.DataFrame({
                            'ds': X_val['Full_Date'],
                            'y': y_val[parameters['target_column']]
                            })
    
    set_random_seed(parameters['random_state'])
    model_metrics = model.fit(df_train, freq='M', validation_df=df_val, batch_size=4, metrics='RMSE', progress='print', epochs=20)
    print('Model_type: ', type(model) )

    model_rmse_val_last_epoch = model_metrics['RMSE_val'].iloc[-1]

    return model, model_metrics, model_rmse_val_last_epoch

def train_fb_prophet(model,
                X_train: pd.DataFrame, 
                X_val: pd.DataFrame, 
                y_train: pd.DataFrame, 
                y_val: pd.DataFrame,
                parameters: Dict[str, Any]):
    """
    """
    df_train = pd.DataFrame({
                        'ds': X_train['Full_Date'],
                        'y': y_train[parameters['target_column']]
                        })

    df_val = pd.DataFrame({
                        'ds': X_val['Full_Date'],
                        'y': y_val[parameters['target_column']]
                        })
    model.fit(df_train)

    # Make future dataframe including validation period
    future = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

    # Forecast
    forecast = model.predict(future)

    # Ensure forecasted and actual values are properly aligned
    forecast_vals = forecast[['yhat']].iloc[-len(df_val):].reset_index(drop=True)
    actual_vals = df_val['y'].reset_index(drop=True)

    # Calculate RMSE
    rmse_val = np.sqrt(((forecast_vals['yhat'] - actual_vals) ** 2).mean())

    results = pd.concat([forecast[['ds', 'yhat']], future], axis=1)

    return model, results, rmse_val