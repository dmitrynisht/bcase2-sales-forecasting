import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np  
import yaml
import pickle
import warnings
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

from .utils import create_actual_vs_predicted_plot
warnings.filterwarnings("ignore", category=Warning)

logger = logging.getLogger(__name__)

     
def model_predict(model,
                    X_train: pd.DataFrame, 
                    X_val: pd.DataFrame, 
                    y_train: pd.DataFrame, 
                    y_val: pd.DataFrame,
                    test_data: pd.DataFrame,
                    parameters: Dict[str, Any]):
    
    
    """Trains a model on the given data and saves it to the given model path.

    Args:
    --  
        model: Production model for prediction.
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): Validation features.
        y_train (pd.DataFrame): Training target.
        y_val (pd.DataFrame): Validation target.
        test_data (pd.DataFrame): Test Data for prediction.
        parameters (dict): Parameters defined in parameters.yml.

    Returns:
    --
        models (dict): Dictionary of trained models.
        scores (pd.DataFrame): Dataframe of model scores.
    """

    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.load(f, Loader=yaml.loader.SafeLoader)['tracking']['experiment']['name']
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        logger.info(experiment_id)

    prediction_results = {}

    logger.info('Starting first step of model prediction.')

    X_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_full = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    df_full = pd.DataFrame({
                            'ds': X_full[parameters['date_column']],
                            'y': y_full[parameters['target_column']]
                            })


    future = model.make_future_dataframe(df_full, n_historic_predictions=True, periods=len(test_data))
    forecast = model.predict(future)

    # Merge the DataFrames on 'full_date' and 'ds'
    merged_df = pd.merge(forecast, test_data, left_on='ds', right_on=parameters['date_column'], how='left')

    # Use 'Sales_EUR' to fill NaN values in 'y'
    merged_df['y'] = merged_df['y'].fillna(merged_df[parameters['target_column']])

    # Drop the 'Sales_EUR' and 'full_date' columns if no longer needed
    merged_df.drop(columns=test_data.columns, inplace=True)

    actual_sales = merged_df['y'].values[-len(test_data):]
    predicted_sales = merged_df['yhat1'].values[-len(test_data):]

    rmse = np.sqrt(mean_squared_error(actual_sales, predicted_sales))
    prediction_results['rmse_test'] = rmse

    fig = create_actual_vs_predicted_plot(merged_df, forecast, df_full, parameters)

    df_predictions = merged_df[['ds', 'y', 'yhat1']].iloc[-len(test_data):].reset_index(drop=True)

    # Dictionary to rename columns
    rename_dict = {
        'ds': 'Date',
        'y': 'Actual_Sales_EUR',
        'yhat1': 'Predicted_Sales_EUR'
    }

    # Rename columns
    df_predictions.rename(columns=rename_dict, inplace=True)

    return df_predictions, prediction_results, fig


