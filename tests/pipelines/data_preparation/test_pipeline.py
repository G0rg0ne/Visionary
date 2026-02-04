"""
Tests for the data_preparation pipeline (Kedro 1.1.1).
"""
import pandas as pd
import pytest

from visionary.pipelines.data_preparation.nodes import split_timeseries_data


def test_split_timeseries_data_per_flight_has_exactly_test_timesteps():
    """Per-flight split: each flight has exactly test_timesteps (9) in test, rest in train."""
    # Two flights, 20 and 18 timesteps each (both >= 8+9=17)
    base = pd.to_datetime("2026-01-01")
    rows = []
    for i, item_id in enumerate(["flight_A", "flight_B"]):
        n = 20 if i == 0 else 18
        for j in range(n):
            rows.append({
                "item_id": item_id,
                "timestamp": base + pd.Timedelta(days=j),
                "target": 100.0 + j,
            })
    df = pd.DataFrame(rows)

    params = {"test_timesteps": 9, "min_train_timesteps": 8}
    train_data, test_data = split_timeseries_data(df, params)

    # Every flight in test must have exactly 9 rows (AutoGluon requires length > prediction_length, i.e. >= 8)
    test_per_item = test_data.groupby("item_id").size()
    assert (test_per_item == 9).all(), f"Expected 9 test timesteps per flight, got {test_per_item.to_dict()}"
    # Train: at least 8 per flight
    train_per_item = train_data.groupby("item_id").size()
    assert (train_per_item >= 8).all(), f"Expected >= 8 train timesteps per flight, got {train_per_item.to_dict()}"
    # flight_A: 20 total -> 11 train, 9 test; flight_B: 18 total -> 9 train, 9 test
    assert len(train_data) == 11 + 9
    assert len(test_data) == 9 + 9
