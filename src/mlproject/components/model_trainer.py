import os
import sys
import joblib
import mlflow
import mlflow.xgboost

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report
)

from src.mlproject.exception import CustomException


class ModelTrainer:

    def __init__(self):
        self.model_path = os.path.join(
            "artifacts",
            "model.pkl"
        )

    def initiate_model_trainer(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        try:

            # MLflow experiment
            mlflow.set_experiment("Customer-Churn-Prediction")

            model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.7,
                scale_pos_weight=2.0,
                random_state=42,
                eval_metric="logloss"
            )

            with mlflow.start_run():

                # Log parameters
                mlflow.log_params({
                    "n_estimators": 100,
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "min_child_weight": 3,
                    "subsample": 0.8,
                    "colsample_bytree": 0.7,
                    "scale_pos_weight": 2.0
                })

                print("Training XGBoost model...")

                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)

                # Probabilities
                y_prob = model.predict_proba(X_test)[:, 1]

                # Metrics
                accuracy = accuracy_score(
                    y_test,
                    y_pred
                )

                roc_auc = roc_auc_score(
                    y_test,
                    y_prob
                )

                print("\nModel training completed")
                print("Accuracy:", accuracy)
                print("ROC-AUC:", roc_auc)

                print("\nClassification Report:")
                print(
                    classification_report(
                        y_test,
                        y_pred
                    )
                )

                # Log metrics
                mlflow.log_metric(
                    "accuracy",
                    accuracy
                )

                mlflow.log_metric(
                    "roc_auc",
                    roc_auc
                )

                # Log model
                mlflow.xgboost.log_model(
                    model,
                    "model"
                )

                # Save model locally
                os.makedirs(
                    "artifacts",
                    exist_ok=True
                )

                joblib.dump(
                    model,
                    self.model_path
                )

                print(
                    "Model saved at:",
                    self.model_path
                )

            return accuracy, roc_auc

        except Exception as e:
            raise CustomException(e, sys)