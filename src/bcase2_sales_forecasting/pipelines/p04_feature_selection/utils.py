import pandas as pd
import re
from typing import Any, Dict


def debug_on_success_(data: pd.DataFrame, dummy_value: int, pipeline_name: str = "", f_verbose: bool = False) -> None:
    
    # Print columns
    if f_verbose:
        print(data.dtypes)

    # dummy_value is for checking pipelines sequence
    dummy_value.append(dummy_value[-1] + 1) 
    print(f"pipeline {pipeline_name} succeed !; f_verbose={f_verbose};", dummy_value)

    return
