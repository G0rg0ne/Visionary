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

def add_holidays(merged_data: pd.DataFrame, airport_country_mapping: dict) -> pd.DataFrame:
    merged_data['origin_country'] = merged_data['origin'].map(airport_country_mapping)
    merged_data['destination_country'] = merged_data['destination'].map(airport_country_mapping)
    merged_data['origin_departure_holidays'] = merged_data.apply(lambda row: CountryHoliday(row['origin_country'], years=row['departure_date'].year).get(row['departure_date']), axis=1)
    merged_data['destination_departure_holidays'] = merged_data.apply(lambda row: CountryHoliday(row['destination_country'], years=row['departure_date'].year).get(row['departure_date']), axis=1)
    return merged_data

def feature_engineering(merged_data: pd.DataFrame, airport_country_mapping: dict) -> pd.DataFrame:
    merged_data = fix_data_types(merged_data)
    merged_data = add_holidays(merged_data, airport_country_mapping)
    return merged_data
