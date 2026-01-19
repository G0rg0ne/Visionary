"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 1.1.1
"""
import pandas as pd
from loguru import logger
from holidays import CountryHoliday
from datetime import datetime

def fix_data_types(merged_data: pd.DataFrame) -> pd.DataFrame:
    merged_data['query_date'] = pd.to_datetime(merged_data['query_date'])
    merged_data['departure_date'] = pd.to_datetime(merged_data['departure_date'])
    return merged_data

def parse_custom_format(series, year_series):
    # Format example: "5:35 PM on Thu, Jan 15" + " 2025"
    # We append the year because Jan 15 exists in every year
    combined_str = series + " " + year_series.astype(str)
    return pd.to_datetime(combined_str, format='%I:%M %p on %a, %b %d %Y', errors='coerce')

def add_temporal_features(merged_data: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from departure_date.
    
    Adds:
    - day_of_week: Day of week as integer (0=Monday, 6=Sunday)
    - is_weekend: Boolean indicating if departure is on weekend (Saturday or Sunday)
    """
    merged_data['departure_day_of_week'] = merged_data['departure_date'].dt.dayofweek
    merged_data['departure_is_weekend'] = merged_data['departure_date'].dt.dayofweek.isin([5, 6])

    merged_data['departure_time_dt'] = parse_custom_format(merged_data['departure_time'], merged_data['departure_date'].dt.year)
    merged_data['arrival_time_dt'] = parse_custom_format(merged_data['arrival_time'], merged_data['departure_date'].dt.year)
    for col in ['departure_time_dt', 'arrival_time_dt']:
        prefix = col.replace('_dt', '')
        merged_data[f'{prefix}_hour'] = merged_data[col].dt.hour
        merged_data[f'{prefix}_minute'] = merged_data[col].dt.minute
    return merged_data

def handle_categorical_features(merged_data: pd.DataFrame) -> pd.DataFrame:
    cat_cols = ['origin', 'destination', 'airline', 'cabin_class', 'arrival_airport']
    for col in cat_cols:
        # Fill missing values with 'None' and ensure type is string
        merged_data[col] = merged_data[col].replace('', None).fillna('None').astype(str)
    return merged_data

def add_holidays(merged_data: pd.DataFrame, airport_country_mapping: dict) -> pd.DataFrame:
    merged_data['origin_country'] = merged_data['origin'].map(airport_country_mapping)
    merged_data['destination_country'] = merged_data['destination'].map(airport_country_mapping)
    merged_data['origin_departure_holidays'] = merged_data.apply(lambda row: CountryHoliday(row['origin_country'], years=row['departure_date'].year).get(row['departure_date']), axis=1)
    merged_data['destination_departure_holidays'] = merged_data.apply(lambda row: CountryHoliday(row['destination_country'], years=row['departure_date'].year).get(row['departure_date']), axis=1)
    return merged_data

def feature_engineering(merged_data: pd.DataFrame, airport_country_mapping: dict) -> pd.DataFrame:
    merged_data = fix_data_types(merged_data)
    merged_data = add_temporal_features(merged_data)
    merged_data = add_holidays(merged_data, airport_country_mapping)
    cols_to_drop = ['departure_time', 'arrival_time', 'currency', 'price', 'source', 'departure_date', 'arrival_date', 'departure_time_dt', 'arrival_time_dt', 'query_date','cabin', 'offer_rank']
    X = merged_data.drop(columns=cols_to_drop)
    y = merged_data['price']
    return merged_data
