"""
This is a boilerplate pipeline 'data_preparation'
generated using Kedro 1.1.1
"""
import pandas as pd
from typing import Dict

# Columns defining a unique flight time series (used for item_id and grouping)
UNIQUE_FLIGHT_COLUMNS = [
    "origin",
    "destination",
    "departure_date",
    "departure_time_dt",
    "airline",
]

def parse_custom_format(series):
    combined_str = series.values.astype('object') 
    parsed = pd.to_datetime(combined_str, format='%I:%M %p on %a, %b %d', errors='coerce')
    return parsed.time

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
    return merged_data

def filter_flight(merged_data: pd.DataFrame, num_samples) -> pd.DataFrame:
    """
    Filter the flights from the merged_data dataframe.
    """
    merged_data['query_date'] = pd.to_datetime(merged_data['query_date'])
    merged_data['departure_date'] = pd.to_datetime(merged_data['departure_date'])

    merged_data['departure_time_dt'] = parse_custom_format(merged_data['departure_time'])
    merged_data['arrival_time_dt'] = parse_custom_format(merged_data['arrival_time'])

    keys = ['origin', 'destination', 'airline','departure_date', 'departure_time_dt', 'arrival_time_dt','flight_duration','days_before_departure']
    merged_data = merged_data.drop_duplicates(subset=keys)

    # Match viz grouping: filter per (route, date, time) so each plot has > num_samples points
    flight_cols = ["origin", "destination", "airline", "departure_date", "departure_time_dt", "arrival_time_dt", "flight_duration"]
    merged_data_df = merged_data[
        merged_data.groupby(flight_cols).transform("size") > num_samples
    ]
    return merged_data_df


def _make_item_id(row: pd.Series) -> str:
    """Build a unique item_id for a flight (safe for AutoGluon)."""
    dep_date = row["departure_date"]
    if hasattr(dep_date, "strftime"):
        dep_date = dep_date.strftime("%Y-%m-%d")
    dep_time = row["departure_time_dt"]
    if hasattr(dep_time, "strftime"):
        dep_time = dep_time.strftime("%H-%M-%S")
    airline_safe = str(row["airline"]).replace(" ", "_")
    return f"{row['origin']}_{row['destination']}_{dep_date}_{dep_time}_{airline_safe}"


def prepare_timeseries_data(
    sampled_dataset: pd.DataFrame, params: Dict
) -> pd.DataFrame:
    """
    Prepare time series data for AutoGluon: fill missing dates per flight and
    forward-fill prices. Output has item_id, timestamp, target and optional
    static/dynamic covariates.
    """
    df = sampled_dataset.copy()
    df["query_date"] = pd.to_datetime(df["query_date"])
    df["departure_date"] = pd.to_datetime(df["departure_date"])

    # One price per (flight, query_date): take best offer_rank per group
    flight_cols = UNIQUE_FLIGHT_COLUMNS + ["query_date"]
    agg_df = (
        df.sort_values("offer_rank")
        .groupby(flight_cols, dropna=False)
        .agg(
            price=("price", "first"),
            flight_duration=("flight_duration", "first"),
            stops=("stops", "first"),
            cabin=("cabin", "first"),
        )
        .reset_index()
    )

    forward_fill_limit = params.get("forward_fill_limit", 3)

    rows_list = []
    for key, group in agg_df.groupby(UNIQUE_FLIGHT_COLUMNS, dropna=False):
        group = group.sort_values("query_date").copy()
        first_row = group.iloc[0]
        origin, dest, dep_date, dep_time, airline = key
        date_min = group["query_date"].min()
        date_max = group["query_date"].max()
        full_dates = pd.date_range(start=date_min, end=date_max, freq="D")
        full_df = pd.DataFrame({"query_date": full_dates})
        merged = full_df.merge(
            group[["query_date", "price"]],
            on="query_date",
            how="left",
        )
        merged["price"] = merged["price"].ffill(limit=forward_fill_limit)
        merged = merged.dropna(subset=["price"])
        merged["origin"] = origin
        merged["destination"] = dest
        merged["departure_date"] = dep_date
        merged["departure_time_dt"] = dep_time
        merged["airline"] = airline
        merged["flight_duration"] = first_row["flight_duration"]
        merged["stops"] = first_row["stops"]
        merged["cabin"] = first_row["cabin"]
        merged["days_before_departure"] = (dep_date - merged["query_date"]).dt.days
        rows_list.append(merged)

    out = pd.concat(rows_list, ignore_index=True)
    out["item_id"] = out.apply(_make_item_id, axis=1)
    out = out.rename(columns={"query_date": "timestamp", "price": "target"})

    # Columns expected by AutoGluon: item_id, timestamp, target; optional static/dynamic
    static_cols = ["origin", "destination", "airline", "flight_duration", "stops", "cabin"]
    keep_cols = ["item_id", "timestamp", "target", "days_before_departure"] + [
        c for c in static_cols if c in out.columns
    ]
    return out[[c for c in keep_cols if c in out.columns]]


def split_timeseries_data(
    timeseries_prepared: pd.DataFrame, params: Dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series into train and test by query_date (timestamp) to avoid
    data leakage: train = all observations with timestamp <= split_date,
    test = all observations with timestamp > split_date.
    """
    df = timeseries_prepared.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    unique_dates = df["timestamp"].dt.normalize().unique()
    unique_dates = sorted(unique_dates)

    test_size = params.get("test_size", 0.2)
    min_test_dates = params.get("min_test_dates", 8)
    n_dates = len(unique_dates)
    n_test_dates = max(1, int(n_dates * test_size))
    n_test_dates = max(n_test_dates, min(min_test_dates, n_dates - 1))
    n_test_dates = min(n_test_dates, n_dates - 1)
    split_idx = n_dates - n_test_dates
    split_date = unique_dates[split_idx]

    train_mask = df["timestamp"].dt.normalize() <= split_date
    test_mask = df["timestamp"].dt.normalize() > split_date

    train_data = df[train_mask].copy()
    test_data = df[test_mask].copy()
    return train_data, test_data