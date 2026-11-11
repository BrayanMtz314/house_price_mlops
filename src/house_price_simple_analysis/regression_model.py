"""Regression model class, which includes methods for training, evaluating, and logging the model to MLflow."""


import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from house_price_simple_analysis.config import ProjectConfig


class RegressionModel:
    """Train, evaluate, and track a house-price regression model."""

    def __init__(self, config: ProjectConfig, spark: SparkSession):
        self.config = config
        self.spark = spark

        self.catalog_name = config.databricks.catalog_name
        self.schema_name = config.databricks.schema_name

        self.features = (
            config.data.num_features + config.data.cat_features
        )
        self.target = config.data.target

        self.model = None
        self.train_set = None
        self.test_set = None
        self.predictions = None
        self.metrics = {}
        self.model_uri = None

    def load_data(self) -> None:
        """Load training and testing data from Unity Catalog."""

        train_table = (
            f"{self.catalog_name}.{self.schema_name}."
            f"{self.config.databricks.train_table}"
        )

        test_table = (
            f"{self.catalog_name}.{self.schema_name}."
            f"{self.config.databricks.test_table}"
        )

        self.train_set = self.spark.table(train_table).toPandas()
        self.test_set = self.spark.table(test_table).toPandas()

        if self.train_set.empty or self.test_set.empty:
            raise ValueError("Training or testing dataset is empty.")

        required_columns = self.features + [self.target]

        for name, df in [
            ("Training", self.train_set),
            ("Testing", self.test_set),
        ]:
            missing = [
                col for col in required_columns
                if col not in df.columns
            ]
            if missing:
                raise ValueError(
                    f"{name} dataset missing columns: {missing}"
                )

        for df in [self.train_set, self.test_set]:
            df[self.config.data.num_features] = (
            df[self.config.data.num_features].astype("float64")
            )

    def prepare_model(self) -> None:
        """Initialize the linear regression model."""

        if self.config.data.cat_features:
            raise ValueError(
                "Categorical features require encoding. "
                "This simple model currently supports numeric features only."
            )

        self.model = LinearRegression(
            fit_intercept=self.config.model.fit_intercept
        )

    def train(self) -> None:
        """Train the model using the training dataset."""

        if self.train_set is None:
            raise ValueError("Load the data before training.")

        if self.model is None:
            raise ValueError("Initialize the model before training.")

        X_train = self.train_set[self.features]
        y_train = self.train_set[self.target]

        self.model.fit(X_train, y_train)

    def evaluate(self) -> dict:
        """Evaluate the model using the testing dataset."""

        if self.test_set is None or self.model is None:
            raise ValueError(
                "Load data and train the model before evaluation."
            )

        if not hasattr(self.model, "coef_"):
            raise ValueError("Train the model before evaluation.")

        X_test = self.test_set[self.features]
        y_test = self.test_set[self.target]

        self.predictions = self.model.predict(X_test)

        mse = mean_squared_error(y_test, self.predictions)

        self.metrics = {
            "mae": float(
                mean_absolute_error(y_test, self.predictions)
            ),
            "rmse": float(np.sqrt(mse)),
            "r2": float(
                r2_score(y_test, self.predictions)
            ),
        }

        return self.metrics

    def log_model(self) -> str:
        """Log the trained model, parameters, and metrics to MLflow."""

        if self.model is None or not hasattr(self.model, "coef_"):
            raise ValueError("Train the model before logging.")

        if not self.metrics:
            raise ValueError("Evaluate the model before logging.")

        mlflow.set_experiment(
            self.config.mlflow.experiment_name
        )

        X_train = self.train_set[self.features]
        input_example = X_train.head(5)

        with mlflow.start_run(run_name="linear_regression_basic"):

            mlflow.log_params({
                "fit_intercept": self.model.fit_intercept,
                "features": ",".join(self.features),
                "target": self.target,
            })

            mlflow.log_metrics(self.metrics)

            model_info = mlflow.sklearn.log_model(
                sk_model=self.model,
                name="house_price_linear_regression",
                input_example=input_example,
            )

            self.model_uri = model_info.model_uri

        return self.model_uri
    
    def register_model(self) -> str:
        """Register the logged model in Unity Catalog."""

        if self.model_uri is None:
            raise ValueError(
                "Log the model before registering it."
            )

        mlflow.set_registry_uri("databricks-uc")

        model_name = (
            f"{self.catalog_name}.{self.schema_name}."
            f"{self.config.model.registered_model_name}"
        )

        registered_model = mlflow.register_model(
            model_uri=self.model_uri,
            name=model_name
        )

        self.registered_model_name = registered_model.name
        self.registered_model_version = registered_model.version

        return (
            f"models:/{registered_model.name}/"
            f"{registered_model.version}"
        )

    def set_model_alias(self, alias: str = "Champion") -> None:
        """Assign an alias to the registered model version."""

        if not hasattr(self, "registered_model_version"):
            raise ValueError("Register the model first.")

        mlflow.set_registry_uri("databricks-uc")

        client = MlflowClient()

        client.set_registered_model_alias(
            name=self.registered_model_name,
            alias=alias,
            version=self.registered_model_version
        )
