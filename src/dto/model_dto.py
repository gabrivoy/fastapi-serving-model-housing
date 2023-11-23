"""
Model DTO.

This module contains the DTO for the model prediction.
"""

import numpy as np
import pandas as pd

from pydantic import BaseModel
from typing import Optional

COLUMN_ALIASES = {
    "uid": "uid",
    "city": "city",
    "description": "description",
    "homeType": "homeType",
    "latitude": "latitude",
    "longitude": "longitude",
    "garage_spaces": "garageSpaces",
    "has_spa": "hasSpa",
    "year_built": "yearBuilt",
    "num_of_patios_and_porch": "numOfPatioAndPorchFeatures",
    "lot_size_sq_ft": "lotSizeSqFt",
    "avg_school_rating": "avgSchoolRating",
    "median_students_per_teacher": "MedianStudentsPerTeacher",
    "num_of_bathrooms": "numOfBathrooms",
    "num_of_bedrooms": "numOfBedrooms"
}

class ModelPredictionDTO(BaseModel):
    """
    Model Prediction DTO.

    This class contains the DTO for the model prediction.
    """

    uid: Optional[int] = None
    city: str
    description: Optional[str] = None
    homeType: str
    latitude: float
    longitude: float
    garage_spaces: int
    has_spa: bool
    year_built: int
    num_of_patios_and_porch: int
    lot_size_sq_ft: float
    avg_school_rating: float
    median_students_per_teacher: float
    num_of_bathrooms: float
    num_of_bedrooms: int

    def as_df(self) -> pd.DataFrame:
        """
        Returns the DTO as a pandas DataFrame.

        Returns:
            pd.DataFrame: DTO as a pandas DataFrame.
        """

        data_dict = {
            COLUMN_ALIASES[key]: value for
            key, value in self.model_dump().items()
        }
        return pd.DataFrame(data_dict, index=[0])