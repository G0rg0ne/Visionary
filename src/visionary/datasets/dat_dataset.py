from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path
from pathlib import PurePosixPath
import fsspec
import pandas as pd

class dat_airport_dataset(AbstractDataset):
    """
    Load and save the airport data from dat file using pandas.
    Example:
        airport_dataset = dat_airport_dataset(filepath="data/03_primary/airports.dat")
        data = airport_dataset.load()
        airport_dataset.save(data)
    """

    def __init__(self, filepath: str):
        super().__init__()
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

    def _describe(self) -> dict:
        return dict(path=str(self._filepath))

    def _load(self) -> pd.DataFrame:
        # Load the .dat file with specified columns for IATA, Lat, Long
        with self._fs.open(str(self._filepath), "rb") as f:
            return pd.read_csv(
                f,
                header=None,
                usecols=[4, 6, 7],
                names=["IATA", "Lat", "Long"],
                na_values="\\N",
            )

    def _save(self, data: pd.DataFrame) -> None:
        with self._fs.open(str(self._filepath), "w") as f:
            f.write(data.to_csv(index=False))
