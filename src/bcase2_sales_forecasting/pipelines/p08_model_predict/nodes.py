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
                    df_full: pd.DataFrame, 
                    df_test: pd.DataFrame,
                    parameters: Dict[str, Any]):
    
    """Trains a model on the given data and saves it to the given model path.

    Args:
    --  
        model: Production model for prediction.
        df_full (pd.DataFrame): Training Data.
        df_test (pd.DataFrame): Test Data for prediction.
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

    class_name = model.__class__.__name__
    prediction_results['Model_Name'] = class_name

    if class_name == 'NeuralProphet':
        future = model.make_future_dataframe(df_full, n_historic_predictions=True, periods=len(df_test))
        forecast = model.predict(future)

        # Convert 'ds' column to datetime in both DataFrames
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        df_test['ds'] = pd.to_datetime(df_test['ds'])

        #  Merge the two DataFrames on the 'ds' column
        merged_df = forecast.merge(df_test, on='ds', how='left', suffixes=('', '_new'))

        # Fill NaN values in 'y' with the corresponding 'y_new' values
        merged_df['y'].fillna(merged_df['y_new'], inplace=True)

        # Drop the 'y_new' column as it is no longer needed
        merged_df.drop(columns=['y_new'], inplace=True)

        actual_sales = merged_df['y'].values[-len(df_test):]
        predicted_sales = merged_df['yhat1'].values[-len(df_test):]

        rmse = np.sqrt(mean_squared_error(actual_sales, predicted_sales))

        df_results = merged_df[['ds', 'y', 'yhat1']]
        df_predictions = merged_df[['ds', 'y', 'yhat1']].iloc[-len(df_test):].reset_index(drop=True)

        # Dictionary to rename columns
        rename_dict = {
                'ds': 'Date',
                'y': 'Actual_Sales_EUR',
                'yhat1': 'Predicted_Sales_EUR'
        }

        prediction_results['RMSE_TEST_SCORE'] = rmse
        # Rename columns
        df_predictions.rename(columns=rename_dict, inplace=True)
        df_results.rename(columns=rename_dict, inplace=True)

        fig = create_actual_vs_predicted_plot(df_results, df_full, parameters)

        return df_predictions, prediction_results, fig
    
    elif class_name == 'Prophet':
         
        future = pd.concat([df_full, df_test], axis=0).reset_index(drop=True)
        forecast = model.predict(future)

        # Convert 'ds' column to datetime in both DataFrames
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        future['ds'] = pd.to_datetime(future['ds'])

        #  Merge the two DataFrames on the 'ds' column
        df_results = forecast[['ds', 'yhat']].merge(future, on='ds', how='left')

        actual_sales =  df_results['y'].values[-len(df_test):]
        predicted_sales = df_results['yhat'].values[-len(df_test):]

        rmse = np.sqrt(mean_squared_error(actual_sales, predicted_sales))

        df_predictions = df_results[['ds', 'y', 'yhat']].iloc[-len(df_test):].reset_index(drop=True)

        # Dictionary to rename columns
        rename_dict = {
                'ds': 'Date',
                'y': 'Actual_Sales_EUR',
                'yhat': 'Predicted_Sales_EUR'
        }
        prediction_results['RMSE_TEST_SCORE'] = rmse

        # Rename columns
        df_predictions.rename(columns=rename_dict, inplace=True)
        df_results.rename(columns=rename_dict, inplace=True)

        fig = create_actual_vs_predicted_plot(df_results, df_full, parameters)

        return df_predictions, prediction_results, fig

    else:
        raise NotImplementedError('Production Model Type is not implemented yet!')


