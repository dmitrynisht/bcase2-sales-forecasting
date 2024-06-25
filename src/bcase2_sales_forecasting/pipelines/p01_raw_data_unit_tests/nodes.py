"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import great_expectations as ge

def unit_test(
    raw_sales: pd.DataFrame,
    raw_market: pd.DataFrame
): 

    pd_sales_ge = ge.from_pandas(raw_sales)
    # print(pd_sales_ge.dtypes)
    assert pd_sales_ge.expect_table_column_count_to_equal(3).success == True
    assert pd_sales_ge.expect_column_values_to_be_of_type("DATE", "object").success == True
    assert pd_sales_ge.expect_column_values_to_be_of_type("Mapped_GCK", "object").success == True
    assert pd_sales_ge.expect_column_values_to_be_of_type("Sales_EUR", "object").success == True
    
    pd_market_ge = ge.from_pandas(raw_market)
    print("raw_market dataset holds columns number:", len(pd_market_ge.columns))
    assert pd_market_ge.expect_table_column_count_to_equal(48).success == True
    
    log = logging.getLogger(__name__)
    log.info("Raw sales and market data passed on the unit data tests")

    return 0