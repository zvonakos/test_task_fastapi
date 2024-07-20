from pydantic import BaseModel
from typing import List

class Prediction(BaseModel):
    class_name: str
    probability: float

class PredictionResponse(BaseModel):
    predictions: List[Prediction]