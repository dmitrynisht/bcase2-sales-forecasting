import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score





def load_data(X_train: pd.DataFrame, y_train: pd.DataFrame) -> tuple:
    # Check and rename the date column
    if 'Full_Date' in X_train.columns:
        X_train.rename(columns={'Full_Date': 'ds'}, inplace=True)
    else:
        raise KeyError("The dataset does not contain a column with date or datetime information.")
    
    # Ensure 'ds' column is of datetime type
    X_train['ds'] = pd.to_datetime(X_train['ds'])
    
    # Convert y_train to a Series
    y_train = y_train.squeeze()
    
    return X_train, y_train


def generate_drift(X: pd.DataFrame, y: pd.Series, drift_factor: float = 0.1) -> tuple:
    """
    Introduce drift by shuffling or modifying a percentage of the data.

    Args:
        X: Features (original data).
        y: Target (original data).
        drift_factor: The proportion of data to modify.

    Returns:
        drifted_X: Features with induced drift.
        drifted_y: Target with induced drift.
    """
    n_samples = int(len(X) * drift_factor)

    # Shuffling a portion of the data to create drift
    shuffled_indices = np.random.permutation(n_samples)
    drifted_X = X.copy()
    drifted_X.iloc[:n_samples] = drifted_X.iloc[shuffled_indices].values

    # Adding noise to numerical columns to create drift
    numerical_columns = drifted_X.select_dtypes(include=[np.number]).columns
    for col in numerical_columns:
        drifted_X[col].iloc[:n_samples] += np.random.normal(0, 0.1, n_samples)

    drifted_y = y.copy()  # Assuming y does not need drift (target remains unchanged)

    return drifted_X, drifted_y


def evaluate_model(X_ref: pd.DataFrame, y_ref: pd.Series, X_drift: pd.DataFrame, y_drift: pd.Series, model) -> dict:
    # Ensure 'y' is in X_ref and X_drift
    X_ref['y'] = y_ref
    X_drift['y'] = y_drift
    
    # Predict on reference data
    future_ref = model.make_future_dataframe(df=X_ref, periods=len(X_ref))
    forecast_ref = model.predict(future_ref)
    y_pred_ref = forecast_ref['Sales_EUR'].values[-len(X_ref):] 
    
    # Predict on drift data
    future_drift = model.make_future_dataframe(df=X_drift, periods=len(X_drift))
    forecast_drift = model.predict(future_drift)
    y_pred_drift = forecast_drift['Sales_EUR'].values[-len(X_drift):]  
    
    # Assuming your task is classification and using accuracy score
    accuracy_ref = accuracy_score(y_ref, np.round(y_pred_ref))
    accuracy_drift = accuracy_score(y_drift, np.round(y_pred_drift))
    
    metrics = {
        'accuracy_ref': accuracy_ref,
        'accuracy_drift': accuracy_drift
    }
    
    return metrics

