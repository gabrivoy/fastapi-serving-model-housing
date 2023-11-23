import json

from fastapi import FastAPI

from dto.model_dto import ModelPredictionDTO
from predict import PredictModel

app = FastAPI()

MODEL_PATH = "src/model/_.pkl"
CLASSES_MAP_PATH = "src/data/classes_map.json"

@app.post("/items/{model_name}")
def predict(model_name: str, data: ModelPredictionDTO):
    """
    Predicts the data given the model.

    Args:
        data (ModelPredictionDTO): Data to predict.
    """

    predict = PredictModel(MODEL_PATH.replace("_", model_name))

    prediction = predict.predict(data)

    with open(CLASSES_MAP_PATH, "r") as f:
        classes_map = json.load(f)

    return {
        # "input_data": data,
        "prediction": classes_map[str(prediction[0])]
    }
