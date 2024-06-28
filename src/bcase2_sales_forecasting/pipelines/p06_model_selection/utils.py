import pandas as pd
from typing import Dict, Any
import numpy as np  
import warnings
from neuralprophet import set_random_seed
import mlflow

warnings.filterwarnings("ignore", category=Warning)


def train_neural_prophet(model,
                    X_train: pd.DataFrame, 
                    X_val: pd.DataFrame, 
                    y_train: pd.DataFrame, 
                    y_val: pd.DataFrame,
                    parameters: Dict[str, Any]):
    """
    """
    df_train = pd.DataFrame({
                            'ds': X_train[parameters['date_column']],
                            'y': y_train[parameters['target_column']]
                            })

    df_val = pd.DataFrame({
                            'ds': X_val[parameters['date_column']],
                            'y': y_val[parameters['target_column']]
                            })
    
     # Define fit parameters
    fit_params = {
        'freq': 'M',
        'batch_size': 4,
        'metrics': 'RMSE',
        'progress': 'print',
        'epochs': 20
    }

    # Log the fit parameters
    mlflow.log_params(fit_params)
    
    set_random_seed(parameters['random_state'])
    model_metrics = model.fit(df_train, freq=fit_params['freq'], validation_df=df_val,
                               batch_size=fit_params['batch_size'], metrics=fit_params['metrics'],
                               progress=fit_params['progress'], epochs=fit_params['epochs'])

    model_rmse_val_last_epoch = model_metrics['RMSE_val'].iloc[-1]

    return model, model_rmse_val_last_epoch, fit_params

def train_fb_prophet(model,
                X_train: pd.DataFrame, 
                X_val: pd.DataFrame, 
                y_train: pd.DataFrame, 
                y_val: pd.DataFrame,
                parameters: Dict[str, Any]):
    """
    """
    df_train = pd.DataFrame({
                        'ds': X_train[parameters['date_column']],
                        'y': y_train[parameters['target_column']]
                        })

    df_val = pd.DataFrame({
                        'ds': X_val[parameters['date_column']],
                        'y': y_val[parameters['target_column']]
                        })
    
    # Define fit parameters
    fit_params = {
    }

    # Log the fit parameters
    mlflow.log_params(fit_params)

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

    return model, rmse_val, fit_params