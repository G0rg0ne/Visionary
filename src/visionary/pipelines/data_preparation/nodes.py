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
    Split time series into train and test per flight (item_id): for each flight,
    take the last test_timesteps rows as test, the rest as train. Ensures every
    flight contributes exactly test_timesteps points to the test set for evaluation.
    """
    df = timeseries_prepared.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    test_timesteps = params.get("test_timesteps", 7)
    min_train_timesteps = params.get("min_train_timesteps", 8)
    min_total = min_train_timesteps + test_timesteps

    train_rows = []
    test_rows = []

    for item_id, group in df.groupby("item_id", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        n = len(group)
        if n < min_total:
            continue
        train_rows.append(group.iloc[: -test_timesteps])
        test_rows.append(group.iloc[-test_timesteps:])

    if not train_rows or not test_rows:
        raise ValueError(
            f"No flights have at least {min_total} timesteps "
            f"(min_train={min_train_timesteps}, test={test_timesteps}). "
            "Increase num_samples or reduce min_train_timesteps/test_timesteps."
        )

    train_data = pd.concat(train_rows, ignore_index=True)
    test_data = pd.concat(test_rows, ignore_index=True)

    return train_data, test_data