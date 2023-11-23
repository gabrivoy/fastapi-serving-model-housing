"""
Module containing the required classes to predict the data.

"""

import pickle

import numpy as np

from sklearn.pipeline import Pipeline
from dto.model_dto import ModelPredictionDTO

Model = Pipeline


class PredictModel():
    """
    PredictModelDTO is a class that loads a model from a given path and predicts
    the given data.
    """

    def __init__(self, model_path: str):
        """ Class constructor. """
        self.model_path = model_path
        self.model: Model = pickle.load(open(self.model_path, 'rb'))

    def predict(self, data: ModelPredictionDTO) -> int:
        """
        Predicts the data given the model.

        Args:
            data (ModelPredictionDTO): Data to predict.

        Returns:
            int: Prediction.
        """
        predictions_df = data.as_df()
        predictions = self.model.predict(predictions_df)
        return predictions
