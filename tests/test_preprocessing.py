import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from src.bcase2_sales_forecasting.pipelines.p03_data_preprocessing.nodes import preprocess_sales

def test_preprocess_sales():
    # Create a sample input DataFrame
    data = pd.DataFrame({
        'full_date': ['01-10-2018', '03-10-2018', '06-08-2019', '03-10-2018'],
        'gck': ['#1', '#3', '#1', "#1"],
        'sales_eur': [100.0, 200.0, 150.0, 60]
    })

    # Define the expected output DataFrame after preprocessing
    # The expecteded output should be a group by month and gck, summing the values
    # of the sales
    expected_output = pd.DataFrame({
        'full_date': ['2018-10', '2018-10', '2019-08'],
        'gck': ['#1', '#3', "#1"],
        'sales_eur': [160.0, 200.0, 150.0]
    })

    expected_output['full_date'] = expected_output['full_date'].astype('period[M]')
    # Define the parameters and dummy value
    parameters = {
        "debug_output": {
            "preprocess_sales": False
        }
    }

    # Call the preprocess_sales function
    output, _ = preprocess_sales(data, parameters, None)

    # Assert that the output DataFrame matches the expected output
    assert expected_output.columns.to_list() == output.columns.to_list()
    assert_frame_equal(output, expected_output)