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
    
    # Merge the dataframes and drop duplicates
    merged_data = pd.concat(dataframes, ignore_index=True)
    keys = ['origin', 'destination', 'airline', 'departure_time', 'departure_date', 'days_before_departure']
    merged_data = merged_data.drop_duplicates(subset=keys)
    return merged_data

def filter_flight(merged_data: pd.DataFrame, num_samples) -> pd.DataFrame:
    """
    Filter the flights from the merged_data dataframe.
    """
    flight_cols = ["origin", "destination", "airline", "departure_time"]
    merged_data_df = merged_data[
        merged_data.groupby(flight_cols)["departure_time"]
        .transform("size") > num_samples
    ]  
    return merged_data_df