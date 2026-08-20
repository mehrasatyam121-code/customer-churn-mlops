import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.mlproject.exception import CustomException


class DataIngestion:

    def __init__(self):
        self.raw_data_path = os.path.join(
            "data",
            "raw",
            "Telco-Customer-Churn.csv"
        )

    def initiate_data_ingestion(self):

        try:
            df = pd.read_csv(self.raw_data_path)

            print("Dataset loaded successfully")
            print("Shape:", df.shape)

            train_set, test_set = train_test_split(
                df,
                test_size=0.20,
                random_state=42,
                stratify=df["Churn"]
            )

            os.makedirs("artifacts", exist_ok=True)

            train_path = os.path.join(
                "artifacts",
                "train.csv"
            )

            test_path = os.path.join(
                "artifacts",
                "test.csv"
            )

            train_set.to_csv(train_path, index=False)
            test_set.to_csv(test_path, index=False)

            print("Train and test files created successfully")
            print("Training shape:", train_set.shape)
            print("Testing shape:", test_set.shape)

            return train_path, test_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    obj = DataIngestion()

    train_path, test_path = obj.initiate_data_ingestion()

    print("Train path:", train_path)
    print("Test path:", test_path)