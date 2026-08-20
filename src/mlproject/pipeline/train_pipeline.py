import sys

from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainer
from src.mlproject.exception import CustomException


def run_training_pipeline():

    try:

        # -------------------------
        # 1. Data Ingestion
        # -------------------------

        data_ingestion = DataIngestion()

        train_path, test_path = (
            data_ingestion.initiate_data_ingestion()
        )

        # -------------------------
        # 2. Data Transformation
        # -------------------------

        data_transformation = DataTransformation()

        (
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor_path
        ) = data_transformation.initiate_data_transformation(
            train_path,
            test_path
        )

        # -------------------------
        # 3. Model Training
        # -------------------------

        model_trainer = ModelTrainer()

        accuracy, roc_auc = (
            model_trainer.initiate_model_trainer(
                X_train,
                X_test,
                y_train,
                y_test
            )
        )

        print("\n==============================")
        print("TRAINING PIPELINE COMPLETED")
        print("==============================")

        print("Accuracy:", accuracy)
        print("ROC-AUC:", roc_auc)

    except Exception as e:

        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()