import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from src.bcase2_sales_forecasting.pipelines.p03_data_preprocessing.nodes import preprocess_sales

def test_preprocess_sales():
    # Create a sample input DataFrame
    data = pd.DataFrame({
        'full_date': ['11-11-2018', '11-11-2018', '03-10-2018', '06-08-2019', '03-10-2018', '01-01-2019', '02-01-2019'],
        'gck': ['#1', '#1', '#3', '#1', "#1", "#3", "#3"],
        'sales_eur': [100.0, 99.0, 200.0, 150.0, 60.0, 10.0, 10.0]
    })
    data['full_date'] = pd.to_datetime(data['full_date'])

    expected_output = pd.DataFrame({
        'full_date': ['2018-03', '2018-03', '2018-11', '2019-01', '2019-02', '2019-06'],
        'gck': ['#1', '#3', "#1", "#3", "#3", "#1"],
        'sales_eur': [60, 200.0, 129.5, 10.0, 10.0, 150.0]
    })

    expected_output['full_date'] = expected_output['full_date'].astype('period[M]')
    expected_output['sales_eur'] = expected_output['sales_eur'].astype('float32')

    # Define the parameters and dummy value
    parameters = {
        "debug_output": {
            "preprocess_sales": False
        }
    }

    # Call the preprocess_sales function
    output = preprocess_sales(data, parameters)

    # Assert that the output DataFrame matches the expected output
    assert expected_output.columns.to_list() == output.columns.to_list()
    assert_frame_equal(output, expected_output)