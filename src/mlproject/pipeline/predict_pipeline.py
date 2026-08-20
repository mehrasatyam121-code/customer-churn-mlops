import sys
import pandas as pd
import mlflow
import mlflow.xgboost
import os

from src.mlproject.exception import CustomException
from src.mlproject.utils.utils import load_object


class PredictPipeline:

   def __init__(self):
    self.model_path = os.path.join(
        "artifacts",
        "model.pkl"
    )

    self.preprocessor_path = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )

   def predict(self, input_data):

    try:
        # Load trained model
        model = load_object(
            file_path=self.model_path
        )

        # Load preprocessing object
        preprocessor = load_object(
            file_path=self.preprocessor_path
        )

        # Transform input data
        input_data_transformed = preprocessor.transform(
            input_data
        )

        # Make prediction
        prediction = model.predict(
            input_data_transformed
        )

        return prediction

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":

    data = {
        "gender": ["Female"],
        "SeniorCitizen": [0],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "tenure": [12],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": ["DSL"],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["Yes"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["No"],
        "StreamingMovies": ["No"],
        "Contract": ["Month-to-month"],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        "MonthlyCharges": [70.5],
        "TotalCharges": [846.0]
    }

    input_df = pd.DataFrame(data)

    obj = PredictPipeline()

    prediction = obj.predict(input_df)

    print("Prediction:", prediction)