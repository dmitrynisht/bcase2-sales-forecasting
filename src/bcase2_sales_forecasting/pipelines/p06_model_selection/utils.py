import pandas as pd
from typing import Dict, Any
import numpy as np  
import warnings
from neuralprophet import set_random_seed
import mlflow

warnings.filterwarnings("ignore", category=Warning)


def train_neural_prophet(model,
                    df_train: pd.DataFrame, 
                    df_val: pd.DataFrame, 
                    parameters: Dict[str, Any]):
    """
    """
    
    # Define fit parameters
    fit_params = parameters['fit_params_NeuralProphet']

    # Log the fit parameters
    mlflow.log_params(fit_params)
    
    set_random_seed(parameters['random_state'])
    model_metrics = model.fit(df_train, freq=fit_params['freq'], validation_df=df_val,
                               batch_size=fit_params['batch_size'], metrics=fit_params['metrics'],
                               progress=fit_params['progress'], epochs=fit_params['epochs'])

    model_rmse_val_last_epoch = model_metrics['RMSE_val'].iloc[-1]

    print('---------')
    print('NP')
    print(model_metrics)
    print('_____________')

    return model, model_rmse_val_last_epoch, fit_params

def train_fb_prophet(model,
                df_train: pd.DataFrame, 
                df_val: pd.DataFrame,
                parameters: Dict[str, Any]):
    """
    """
    # Define fit parameters
    fit_params = parameters['fit_params_Prophet']

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

    print('---------')
    print('FACEBOOK')
    print(results)
    print('_____________')

    return model, rmse_val, fit_params