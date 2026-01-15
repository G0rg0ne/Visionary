"""
This is a boilerplate pipeline 'data_viz'
generated using Kedro 1.1.1
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterator, Tuple

class FigureGenerator:
    """
    Wrapper class to ensure generator always yields (key, figure) tuples.
    """
    def __init__(self, merged_data: pd.DataFrame):
        self.merged_data = merged_data
    
    def __iter__(self):
        return self._generate()
    
    def _generate(self):
        # Convert date columns to datetime
        merged_data = self.merged_data.copy()
        merged_data['query_date'] = pd.to_datetime(merged_data['query_date'])
        merged_data['departure_date'] = pd.to_datetime(merged_data['departure_date'])
        
        # Group by unique flight
        flight_groups = merged_data.groupby(['origin', 'destination', 'departure_date', 'departure_time'])
        
        # Create and yield a figure for each unique flight
        for group_key, group in flight_groups:
            # Unpack the groupby key safely
            if isinstance(group_key, tuple) and len(group_key) == 4:
                origin, dest, dep_date, dep_time = group_key
            else:
                raise ValueError(f"Unexpected groupby key structure: {group_key} (type: {type(group_key)})")
            
            # Sort by days_before_departure to show evolution as departure approaches
            group_sorted = group.sort_values('days_before_departure')
            
            # Skip groups with fewer than 4 data points
            if len(group_sorted) < 4:
                continue
            
            # Create a unique flight identifier for the dictionary key
            flight_id = f"{origin}_{dest}_{dep_date.strftime('%Y-%m-%d')}_{dep_time.replace(':', '-').replace(' ', '_')}"
            
            # Create figure for this flight
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot price vs days_before_departure
            ax.plot(
                group_sorted['days_before_departure'], 
                group_sorted['price'],
                marker='o',
                linewidth=2,
                markersize=6
            )
            
            # Formatting
            ax.set_xlabel('Days Before Departure', fontsize=12)
            ax.set_ylabel('Price', fontsize=12)
            ax.set_title(f'Price Evolution: {origin} → {dest} | {dep_date.strftime("%Y-%m-%d")} {dep_time}', 
                         fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            # Reverse x-axis to show evolution from oldest to newest (approaching departure)
            ax.invert_xaxis()
            
            plt.tight_layout()
            
            # Yield as tuple - ensure it's always a tuple
            yield (flight_id, fig)

def _generate_figures(merged_data: pd.DataFrame):
    """
    Internal generator function that yields (flight_id, figure) tuples.
    """
    # Convert date columns to datetime
    merged_data = merged_data.copy()
    merged_data['query_date'] = pd.to_datetime(merged_data['query_date'])
    merged_data['departure_date'] = pd.to_datetime(merged_data['departure_date'])
    
    # Group by unique flight
    flight_groups = merged_data.groupby(['origin', 'destination', 'departure_date', 'departure_time'])
    
    # Create and yield a figure for each unique flight
    for group_key, group in flight_groups:
        # Unpack the groupby key safely
        if isinstance(group_key, tuple) and len(group_key) == 4:
            origin, dest, dep_date, dep_time = group_key
        else:
            raise ValueError(f"Unexpected groupby key structure: {group_key} (type: {type(group_key)})")
        
        # Sort by days_before_departure to show evolution as departure approaches
        group_sorted = group.sort_values('days_before_departure')
        
        # Create a unique flight identifier for the dictionary key
        flight_id = f"{origin}_{dest}_{dep_date.strftime('%Y-%m-%d')}_{dep_time.replace(':', '-').replace(' ', '_')}"
        
        # Create figure for this flight
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot price vs days_before_departure
        ax.plot(
            group_sorted['days_before_departure'], 
            group_sorted['price'],
            marker='o',
            linewidth=2,
            markersize=6
        )
        
        # Formatting
        ax.set_xlabel('Days Before Departure', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.set_title(f'Price Evolution: {origin} → {dest} | {dep_date.strftime("%Y-%m-%d")} {dep_time}', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # Reverse x-axis to show evolution from oldest to newest (approaching departure)
        ax.invert_xaxis()
        
        plt.tight_layout()
        
        # Yield as tuple - be very explicit
        yield (flight_id, fig)

def plot_ticket_evolution(merged_data: pd.DataFrame) -> Iterator[Tuple[str, plt.Figure]]:
    """
    Plot the ticket evolution for unique flights.
    Groups by origin, destination, departure_date, and departure_time
    to identify unique flights and tracks price changes over time.
    
    Returns a generator that yields figures one at a time to prevent OOM issues.
    Each figure is yielded as a (flight_id, figure) tuple.
    
    Returns:
        Iterator[Tuple[str, plt.Figure]]: Generator yielding (flight_id, figure) tuples
    """
    # Return the generator wrapper to ensure proper tuple yielding
    return FigureGenerator(merged_data)