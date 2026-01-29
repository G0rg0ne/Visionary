"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 1.1.1
"""
import pandas as pd
from loguru import logger
from holidays import CountryHoliday
from datetime import datetime
import numpy as np
import random
import time
from haversine import haversine
random.seed(42)


def fix_data_types(merged_data: pd.DataFrame) -> pd.DataFrame:
    merged_data['query_date'] = pd.to_datetime(merged_data['query_date'])
    merged_data['departure_date'] = pd.to_datetime(merged_data['departure_date'])
    return merged_data

def parse_custom_format(series, year_series):


    year_str = year_series.astype(str).values
    combined_str = series.values.astype('object') + " " + year_str
    return pd.to_datetime(combined_str, format='%I:%M %p on %a, %b %d %Y', errors='coerce')

def add_temporal_features(merged_data: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from departure_date.
    
    Adds:
    - day_of_week: Day of week as integer (0=Monday, 6=Sunday)
    - is_weekend: Boolean indicating if departure is on weekend (Saturday or Sunday)
    """
    # Cache datetime accessors to avoid repeated computations
    departure_dt = merged_data['departure_date'].dt
    
    merged_data['departure_day_of_week'] = departure_dt.dayofweek
    merged_data['departure_is_weekend'] = merged_data['departure_day_of_week'].isin([5, 6])
    merged_data['departure_week_of_year'] = departure_dt.isocalendar().week
    merged_data['departure_month'] = departure_dt.month

    # Vectorized cyclic encoding
    merged_data['cyclic_sin_departure_day_of_week'] = np.sin(2 * np.pi * merged_data['departure_day_of_week'] / 7)
    merged_data['cyclic_cos_departure_day_of_week'] = np.cos(2 * np.pi * merged_data['departure_day_of_week'] / 7)
    merged_data['cyclic_sin_departure_week_of_year'] = np.sin(2 * np.pi * merged_data['departure_week_of_year'] / 52)
    merged_data['cyclic_cos_departure_week_of_year'] = np.cos(2 * np.pi * merged_data['departure_week_of_year'] / 52)
    merged_data['cyclic_sin_departure_month'] = np.sin(2 * np.pi * merged_data['departure_month'] / 12)
    merged_data['cyclic_cos_departure_month'] = np.cos(2 * np.pi * merged_data['departure_month'] / 12)

    # Parse text defined time to datetime
    departure_year = departure_dt.year
    merged_data['departure_time_dt'] = parse_custom_format(merged_data['departure_time'], departure_year)
    merged_data['arrival_time_dt'] = parse_custom_format(merged_data['arrival_time'], departure_year)
    
    # Vectorized time feature extraction
    departure_time_dt = merged_data['departure_time_dt'].dt
    arrival_time_dt = merged_data['arrival_time_dt'].dt
    
    merged_data['departure_time_day'] = departure_time_dt.day
    merged_data['departure_time_month'] = departure_time_dt.month
    merged_data['departure_time_hour'] = departure_time_dt.hour
    merged_data['departure_time_minute'] = departure_time_dt.minute
    
    merged_data['arrival_time_day'] = arrival_time_dt.day
    merged_data['arrival_time_month'] = arrival_time_dt.month
    merged_data['arrival_time_hour'] = arrival_time_dt.hour
    merged_data['arrival_time_minute'] = arrival_time_dt.minute

    # Use vectorized comparison
    merged_data["overnight_flight"] = departure_time_dt.date < arrival_time_dt.date
    
    # Vectorized cyclic encoding for time features
    merged_data['cyclic_sin_departure_time_hour'] = np.sin(2 * np.pi * merged_data['departure_time_hour'] / 24)
    merged_data['cyclic_cos_departure_time_hour'] = np.cos(2 * np.pi * merged_data['departure_time_hour'] / 24)
    merged_data['cyclic_sin_departure_time_minute'] = np.sin(2 * np.pi * merged_data['departure_time_minute'] / 60)
    merged_data['cyclic_cos_departure_time_minute'] = np.cos(2 * np.pi * merged_data['departure_time_minute'] / 60)
    merged_data['cyclic_sin_arrival_time_hour'] = np.sin(2 * np.pi * merged_data['arrival_time_hour'] / 24)
    merged_data['cyclic_cos_arrival_time_hour'] = np.cos(2 * np.pi * merged_data['arrival_time_hour'] / 24)
    merged_data['cyclic_sin_arrival_time_minute'] = np.sin(2 * np.pi * merged_data['arrival_time_minute'] / 60)
    merged_data['cyclic_cos_arrival_time_minute'] = np.cos(2 * np.pi * merged_data['arrival_time_minute'] / 60)
    
    return merged_data

def handle_categorical_features(merged_data: pd.DataFrame) -> pd.DataFrame:
    merged_data['departure_is_weekend'] = merged_data['departure_is_weekend'].astype(bool)
    cat_cols = ['origin', 'destination', 'airline','origin_country','destination_country']
    for col in cat_cols:
        merged_data[col] = merged_data[col].fillna('None').astype(str)
    return merged_data

def add_holidays(merged_data: pd.DataFrame, airport_country_mapping: dict) -> pd.DataFrame:
    """Optimized holiday feature engineering using pre-computed holiday sets."""
    
    merged_data['origin_country'] = merged_data['origin'].map(airport_country_mapping)
    merged_data['destination_country'] = merged_data['destination'].map(airport_country_mapping)
    
    # Pre-compute all unique countries and years to minimize CountryHoliday object creation
    unique_countries = set(merged_data['origin_country'].dropna().unique()) | set(merged_data['destination_country'].dropna().unique())
    unique_years = set(merged_data['departure_date'].dt.year.unique())
    
    logger.info(f"Pre-computing holidays for {len(unique_countries)} countries and {len(unique_years)} years...")
    
    # Create a dictionary of holiday sets for each country-year combination
    holiday_cache = {}
    for country in unique_countries:
        if pd.isna(country) or country == 'None':
            continue
        for year in unique_years:
            try:
                holidays = CountryHoliday(country, years=year)
                holiday_cache[(country, year)] = set(holidays.keys())
            except Exception as e:
                logger.warning(f"Could not load holidays for {country} in {year}: {e}")
                holiday_cache[(country, year)] = set()
    
    # Vectorized holiday checking function
    def get_holiday_name_vectorized(dates, countries):
        """Get holiday names for dates and countries using pre-computed cache."""
        result = []
        for date, country in zip(dates, countries):
            if pd.isna(date) or pd.isna(country) or country == 'None':
                result.append(None)
                continue
            year = pd.Timestamp(date).year
            holidays_set = holiday_cache.get((country, year), set())
            
            # Check if date is in holidays
            date_only = date.date() if hasattr(date, 'date') else date
            if date_only in holidays_set:
                # Get the actual holiday name
                try:
                    holiday_obj = CountryHoliday(country, years=year)
                    result.append(holiday_obj.get(date_only))
                except:
                    result.append('Holiday')
            else:
                result.append(None)
        
        return result
    
    def check_holiday_in_next_7days_vectorized(dates, countries):
        """Check if there's a holiday in next 7 days using pre-computed cache."""
        result = []
        for date, country in zip(dates, countries):
            if pd.isna(date) or pd.isna(country) or country == 'None':
                result.append(False)
                continue
            
            year = pd.Timestamp(date).year
            next_date = date + pd.Timedelta(days=7)
            
            # Get holidays for current year and next year if crossing boundary
            years_needed = {year}
            if next_date.year != year:
                years_needed.add(next_date.year)
            
            # Combine all relevant holidays
            all_holidays = set()
            for y in years_needed:
                all_holidays.update(holiday_cache.get((country, y), set()))
            
            # Check each of the next 7 days
            has_holiday = False
            for j in range(1, 8):
                check_date = (date + pd.Timedelta(days=j)).date()
                if check_date in all_holidays:
                    has_holiday = True
                    break
            
            result.append(has_holiday)
        
        return result
    
    logger.info("Computing holiday features...")
    
    # Apply vectorized functions
    merged_data['origin_departure_holidays'] = get_holiday_name_vectorized(
        merged_data['departure_date'].values, 
        merged_data['origin_country'].values
    )
    
    merged_data['destination_departure_holidays'] = get_holiday_name_vectorized(
        merged_data['departure_date'].values, 
        merged_data['destination_country'].values
    )
    
    merged_data['origin_is_holiday_next_7days'] = check_holiday_in_next_7days_vectorized(
        merged_data['departure_date'].values,
        merged_data['origin_country'].values
    )
    
    merged_data['destination_is_holiday_next_7days'] = check_holiday_in_next_7days_vectorized(
        merged_data['departure_date'].values,
        merged_data['destination_country'].values
    )
    
    logger.info("Holiday features computed successfully")
    
    return merged_data

def handle_categorical_features(merged_data: pd.DataFrame) -> pd.DataFrame:
    cat_cols = ['origin', 'destination', 'airline', 'origin_country', 'destination_country',"origin_departure_holidays","destination_departure_holidays"]
    for col in cat_cols:
        # Fill missing values with 'None' and ensure type is string
        merged_data[col] = merged_data[col].replace('', None).fillna('None').astype(str)
    return merged_data


def feature_engineering(merged_data: pd.DataFrame, airport_country_mapping: dict) -> pd.DataFrame:
    merged_data = fix_data_types(merged_data)
    merged_data = add_temporal_features(merged_data)
    merged_data = add_holidays(merged_data, airport_country_mapping)
    merged_data = handle_categorical_features(merged_data)
    return merged_data

################## data augmentation ##################
def compute_distance_between_airports(od_pairs: set, airport_data: pd.DataFrame) -> dict:
    distance_dict = {}
    for od_pair in od_pairs:
        origin, destination = od_pair
        origin_lat, origin_long = airport_data.loc[airport_data["IATA"] == origin, ["Lat", "Long"]].values[0]
        destination_lat, destination_long = airport_data.loc[airport_data["IATA"] == destination, ["Lat", "Long"]].values[0]
        distance = haversine((origin_lat, origin_long), (destination_lat, destination_long))
        distance_dict[od_pair] = distance
    return distance_dict
def data_augmentation(merged_data: pd.DataFrame, airport_data: pd.DataFrame) -> pd.DataFrame:
    merged_data["od_pairs"] = list(zip(merged_data["origin"], merged_data["destination"]))
    od_pairs = set(merged_data["od_pairs"])
    result_dict = compute_distance_between_airports(od_pairs,airport_data)
    merged_data["airports_distance"] = merged_data["od_pairs"].map(result_dict)
    merged_data = merged_data.drop(columns=["od_pairs"])
    import pdb;pdb.set_trace()
    return merged_data

##################### split data #####################
def split_data(merged_data: pd.DataFrame) -> pd.DataFrame:
    
    merged_data = merged_data.sort_values('query_date').reset_index(drop=True)
    split_idx = int(len(merged_data) * 0.8)
    split_date = merged_data.iloc[split_idx]['query_date']
    # Split chronologically
    train_data = merged_data[merged_data['query_date'] < split_date].copy()
    test_data = merged_data[merged_data['query_date'] >= split_date].copy()
    logger.info(f"Train/Test split date: {split_date}")
    logger.info(f"Train set: {len(train_data)} rows (query_date < {split_date})")
    logger.info(f"Test set: {len(test_data)} rows (query_date >= {split_date})")
    logger.info(f"Train date range: {train_data['query_date'].min()} to {train_data['query_date'].max()}")
    logger.info(f"Test date range: {test_data['query_date'].min()} to {test_data['query_date'].max()}")
    # Drop columns that shouldn't be in features (if not already dropped)
    cols_to_drop = ['departure_time', 'arrival_time', 'currency', 'source', 
                    'departure_date', 'departure_time_dt', 'arrival_time_dt', 
                    'query_date', 'cabin', 'offer_rank','departure_day_of_week','departure_time_minute',
                    'departure_time_hour','arrival_time_minute','arrival_time_hour'
                    ]
    # Only drop columns that exist
    cols_to_drop = [col for col in cols_to_drop if col in train_data.columns]
    
    train_data = train_data.drop(columns=cols_to_drop)
    test_data = test_data.drop(columns=cols_to_drop)
    
    return train_data, test_data


