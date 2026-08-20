import os
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.mlproject.exception import CustomException
from src.mlproject.utils.utils import save_object


class DataTransformation:

    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join(
            "artifacts",
            "preprocessor.pkl"
        )

    def get_data_transformer_object(self):

        try:
            numerical_columns = [
                "SeniorCitizen",
                "tenure",
                "MonthlyCharges",
                "TotalCharges"
            ]

            categorical_columns = [
                "gender",
                "Partner",
                "Dependents",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod"
            ]

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num",
                        StandardScaler(),
                        numerical_columns
                    ),
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore"),
                        categorical_columns
                    )
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Convert TotalCharges to numeric
            train_df["TotalCharges"] = pd.to_numeric(
                train_df["TotalCharges"],
                errors="coerce"
            )

            test_df["TotalCharges"] = pd.to_numeric(
                test_df["TotalCharges"],
                errors="coerce"
            )

            # Fill missing TotalCharges using training median
            train_median = train_df["TotalCharges"].median()

            train_df["TotalCharges"] = train_df["TotalCharges"].fillna(
                train_median
            )

            test_df["TotalCharges"] = test_df["TotalCharges"].fillna(
                train_median
            )

            # Target column
            target_column = "Churn"

            # Convert Yes/No to 1/0
            train_df[target_column] = train_df[target_column].map(
                {"Yes": 1, "No": 0}
            )

            test_df[target_column] = test_df[target_column].map(
                {"Yes": 1, "No": 0}
            )

            # Separate features and target
            X_train = train_df.drop(
                columns=["customerID", target_column]
            )

            y_train = train_df[target_column]

            X_test = test_df.drop(
                columns=["customerID", target_column]
            )

            y_test = test_df[target_column]

            # Create preprocessing object
            preprocessor = self.get_data_transformer_object()

            # Fit on training data
            X_train_processed = preprocessor.fit_transform(X_train)

            # Transform test data
            X_test_processed = preprocessor.transform(X_test)

            # Save preprocessing object
            save_object(
                file_path=self.preprocessor_obj_file_path,
                obj=preprocessor
            )

            print("Data transformation completed")
            print("Training shape:", X_train_processed.shape)
            print("Testing shape:", X_test_processed.shape)

            return (
                X_train_processed,
                X_test_processed,
                y_train,
                y_test,
                self.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)