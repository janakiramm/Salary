from typing import Any
from cog import BasePredictor, Input, Path
import joblib

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model=joblib.load("model.pkl")

    # The arguments and types the model takes as input
    def predict(self,
          exp: int = Input(description="Experience in years")) -> Any:
        """Run a single prediction on the model"""
        sal=self.model.predict([[exp]])
        return round(sal[0],2)