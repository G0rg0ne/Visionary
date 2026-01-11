"""
Custom dataset to handle generator-based figure saving for memory efficiency.
This allows figures to be saved one at a time as they're generated, preventing OOM issues.
"""

from pathlib import Path
from typing import Iterator, Tuple
import matplotlib.pyplot as plt
from kedro.io import AbstractDataset


class GeneratorFigureDataset(AbstractDataset):
    """
    Dataset that accepts a generator of (key, figure) tuples and saves each figure
    immediately to prevent memory issues when processing large numbers of figures.
    
    The generator should yield tuples of (partition_key, matplotlib.Figure).
    Each figure is saved immediately as it's yielded.
    
    Example:
        def generate_figures():
            for key, fig in my_generator():
                yield key, fig
        
        dataset = GeneratorFigureDataset(
            path="data/08_reporting/figures",
            filename_suffix=".png"
        )
        dataset.save(generate_figures())
    """
    
    def __init__(self, path: str, filename_suffix: str = ".png"):
        """
        Initialize the dataset.
        
        Args:
            path: Directory path where figures will be saved
            filename_suffix: Suffix for saved files (e.g., ".png")
        """
        self._path = Path(path)
        self._filename_suffix = filename_suffix
        
    def _describe(self) -> dict:
        """Describe the dataset."""
        return dict(path=str(self._path), filename_suffix=self._filename_suffix)
    
    def _load(self) -> dict:
        """
        Load is not supported for this dataset as it's write-only.
        """
        raise NotImplementedError("Loading is not supported for GeneratorFigureDataset")
    
    def _save(self, data) -> None:
        """
        Save figures from a generator, saving each one immediately.
        
        Args:
            data: Generator yielding (partition_key, matplotlib.Figure) tuples
        """
        # Ensure the directory exists
        self._path.mkdir(parents=True, exist_ok=True)
        
        # Check if data is actually a generator/iterator
        if not hasattr(data, '__iter__'):
            raise TypeError(f"Expected an iterable (generator), but got {type(data)}")
        
        # If data is an iterable class (like our FigureGenerator), get its iterator
        if hasattr(data, '__iter__') and not hasattr(data, '__next__'):
            # It's an iterable object, get its iterator
            data = iter(data)
        
        # Iterate through the generator and save each figure immediately
        for item in data:
            # The generator should yield tuples of (key, figure)
            # Check the type first to provide better error messages
            if isinstance(item, str):
                raise ValueError(
                    f"Generator yielded a string '{item[:50]}...' instead of a tuple (key, figure). "
                    f"This suggests the node function is not yielding tuples correctly. "
                    f"Please check that the node function uses 'yield (key, figure)' not 'yield key'."
                )
            
            # Handle the unpacking with better error messages
            try:
                # Check if it's a tuple or can be unpacked
                if not isinstance(item, tuple):
                    # Try to convert to tuple if it's iterable
                    if hasattr(item, '__iter__') and not isinstance(item, (str, bytes)):
                        item = tuple(item)
                    else:
                        raise TypeError(f"Item is not a tuple and cannot be converted: {type(item)}")
                
                # Verify it has exactly 2 elements
                if len(item) != 2:
                    raise ValueError(
                        f"Expected tuple of length 2 (key, figure), but got length {len(item)}. "
                        f"Item: {item}"
                    )
                
                # Unpack the tuple
                partition_key, figure = item
                
            except (ValueError, TypeError) as e:
                # Provide detailed error information
                item_type = type(item).__name__
                item_repr = repr(item)
                if len(item_repr) > 200:
                    item_repr = item_repr[:200] + "..."
                
                raise ValueError(
                    f"Failed to unpack generator item. Expected (key, figure) tuple, "
                    f"but got {item_type}. Item: {item_repr}. "
                    f"Original error: {str(e)}"
                ) from e
            
            # Construct the full file path
            filename = f"{partition_key}{self._filename_suffix}"
            filepath = self._path / filename
            
            # Save the figure
            figure.savefig(filepath, dpi=100, bbox_inches='tight')
            
            # Close the figure to free memory immediately
            plt.close(figure)
