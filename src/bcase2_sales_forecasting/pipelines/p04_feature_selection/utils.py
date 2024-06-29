import pandas as pd
import numpy as np
import re
import logging
from typing import Any, Dict
from scipy.stats import shapiro
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

def debug_on_success_(data: pd.DataFrame, dummy_value: int, pipeline_name: str = "", f_verbose: bool = False) -> None:
    
    # Print columns
    if f_verbose:
        print(data.dtypes)

    # dummy_value is for checking pipelines sequence
    dummy_value.append(dummy_value[-1] + 1) 
    print(f"pipeline {pipeline_name} succeed !; f_verbose={f_verbose};", dummy_value)

    return


def check_normality(data):
    """Function to check normality using Shapiro-Wilk test
    """
    _, p_value = shapiro(data)
    
    return p_value > 0.05  # Null hypothesis: data is normally distributed if p_value > 0.05


def get_highly_correlated_features(data):
    # Calculate the Spearman correlation matrix
    corr = data.corr(method='spearman')
        
    # Getting correlations with sales (the target)
    sales_correlation = corr['sales_eur'].abs()
        
    # Filtering features with correlation >= 0.4
    highly_correlated_features = sales_correlation[sales_correlation >= 0.4].index.tolist()
        
    # Remove sales from the list if present
    if 'sales_eur' in highly_correlated_features:
        highly_correlated_features.remove('sales_eur')
    
    # Choosing the less correlated feature if multiple features are highly correlated
    chosen_features = []
    for feature in highly_correlated_features:
        correlated_with_sales = corr[feature].abs().drop(index=feature)
        less_correlated_feature = correlated_with_sales.idxmax()
        chosen_features.append(less_correlated_feature)

    logger.info(f"Correlated Features: {chosen_features}\n")
    
    return chosen_features

def get_top_10_features(data):
    # fit XGBoost model and calc feature importances
    # Fit model on training data
    model = XGBRegressor(random_state=1)
    model.fit(data.iloc[:,1:], data.iloc[:,0])  # Sales € is the first column, so it's used as the dependent variable

    # Feature importance
    importance = pd.DataFrame(index=data.iloc[:,1:].columns, data=model.feature_importances_)
    importance.sort_values(by=0, inplace=True)

    # Select the top 10 rows
    top_10 = importance.iloc[-10:]
    
    # Extract the top feature names into a list
    top_feature_list = top_10.index.tolist()

    # Print the list of correlated features
    logger.info(f"Top 10 Features: {top_feature_list}\n")
    
    return top_feature_list

def find_lag(product_code, market_data, product_sales_map, mkt_lagged_datasets):
    """ 
    This function identifies the most significant lag for each macroeconomic index with respect to a given product group's sales data. 
    
    Args:
        product_code (int): The code corresponding to the product group for which the analysis is performed.
        
    Returns:
        pandas.DataFrame: A DataFrame containing the original sales data along with the most important lag features 
        for each macroeconomic index identified through Spearman correlation analysis.
    """

    # Select sales data based on product code
    sales_data = product_sales_map["sales_" + str(product_code)].reset_index(drop=True)

    # Initialize DataFrame to store lag features
    data_with_lags = sales_data.copy()
    
    # List to store the most important lag for each feature
    most_important = [] 

    # Loop through each feature in the market_copy DataFrame
    for feature in market_data.columns:
        # Initialize DataFrame to store lag features for the current feature
        lag_features = pd.DataFrame()
        
        # Create lag features for the current feature
        for lag in range(1, 13):
            lag_feature = mkt_lagged_datasets[f"market_lag{lag}"][feature].rename(f"{feature} LAG{lag}")
            lag_features = pd.concat([lag_features, lag_feature], axis=1)
        
        # Concatenate lag features with sales data
        df_lags = pd.concat([lag_features, sales_data['sales_eur']], axis=1)
        
        # Calculate Spearman correlation matrix
        cor_spearman = df_lags.corr(method='spearman')
        cor_spearman = cor_spearman.mask(np.triu(np.ones(cor_spearman.shape)).astype(bool))
        
        # Find the most important lag for the current feature
        most_important_lag = abs(cor_spearman.loc['sales_eur', :]).idxmax()
        most_important.append(most_important_lag)
        
        # Add the most important lag feature to the DataFrame with lags
        data_with_lags = pd.concat([data_with_lags, df_lags[most_important_lag]], axis=1)
    
    return data_with_lags

def add_sales_lags(lag_data):
    # Find relevant lags
    relevant_lags = []
    for lag in range(1, 7):
        corr = lag_data['sales_eur'].autocorr(lag)
        if abs(corr) > 0.1:
            relevant_lags.append(lag)
    
    # Add relevant lags to lag_data
    for lag in relevant_lags:
        lag_data[f'sales_eur_LAG{lag}'] = lag_data['sales_eur'].shift(lag)
    
    return lag_data
  
