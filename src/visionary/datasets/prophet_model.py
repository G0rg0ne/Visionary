
#make a ProphetModelDataset to use in in data catalog to save and load the model

from prophet import Prophet
from kedro.io import AbstractDataset
import joblib

class ProphetModelDataset(AbstractDataset):
    """
    Dataset to save the Prophet model.
    
    Example:   (filepath='/models/model.joblib')
        model = ProphetModelDataset(filepath='/models/model.joblib')
        model.save(model)
        model = model.load()
    """
    def __init__(self, filepath: str):
        """
        Initialize the dataset.
        """
        self.filepath = filepath

    def _describe(self) -> dict:
        """
        Describe the dataset.
        """
        return dict(filepath=self.filepath)

    def _load(self):
        """
        Load the Prophet model.
        """
        return joblib.load(self.filepath)

    def _save(self, model: Prophet):
        """
        Save the Prophet model.
        """
        joblib.dump(model, self.filepath)