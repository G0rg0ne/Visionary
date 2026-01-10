"""
This is a boilerplate pipeline 'data_preparation'
generated using Kedro 1.1.1
"""
import pandas as pd
from typing import Dict

def merge_dataframes(raw_csv_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge the dataframes from the raw_csv_data dictionary.
    """
    dataframes = []
    for key, dataset in raw_csv_data.items():
        if not key.endswith('.csv'):
            continue
        df = dataset()
        dataframes.append(df)
    
    if not dataframes:
        raise ValueError("No valid CSV DataFrames found in raw_csv_data")
    
    # Merge the dataframes
    merged_data = pd.concat(dataframes, ignore_index=True)
    return merged_data