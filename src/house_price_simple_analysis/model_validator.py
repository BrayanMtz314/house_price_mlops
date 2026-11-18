
import mlflow
import mlflow.sklearn
import numpy as np

from pyspark.sql import SparkSession
from sklearn.metrics import mean_squared_error

from house_price_simple_analysis.config import ProjectConfig


class ModelValidator:
    """Compare a candidate model against the current Champion."""

    def __init__(
        self,
        config: ProjectConfig,
        spark: SparkSession,
        candidate_uri: str,
        min_improvement: float = 0.01,
    ):
        if not 0 <= min_improvement < 1:
            raise ValueError(
                "min_improvement must be between 0 and 1."
            )

        self.config = config
        self.spark = spark
        self.candidate_uri = candidate_uri
        self.min_improvement = min_improvement

        self.model_name = (
            f"{config.databricks.catalog_name}."
            f"{config.databricks.schema_name}."
            f"{config.model.registered_model_name}"
        )

        self.champion_uri = (
            f"models:/{self.model_name}@Champion"
        )

        self.features = (
            config.data.num_features
            + config.data.cat_features
        )
        self.target = config.data.target

        mlflow.set_registry_uri("databricks-uc")

    def load_test_data(self):
        """Load the evaluation dataset from Unity Catalog."""

        table_name = (
            f"{self.config.databricks.catalog_name}."
            f"{self.config.databricks.schema_name}."
            f"{self.config.databricks.test_table}"
        )

        df = self.spark.table(table_name).toPandas()

        if df.empty:
            raise ValueError("Evaluation dataset is empty.")

        X = df[self.features].copy()
        X[self.config.data.num_features] = (
            X[self.config.data.num_features].astype("float64")
        )

        y = df[self.target]

        return X, y

    def evaluate_model(self, model_uri, X, y):
        """Calculate RMSE for a registered or logged model."""

        model = mlflow.sklearn.load_model(model_uri)
        predictions = model.predict(X)

        return float(
            np.sqrt(mean_squared_error(y, predictions))
        )

    def validate(self):
        """Compare Candidate and Champion performance."""

        X_test, y_test = self.load_test_data()

        champion_rmse = self.evaluate_model(
            self.champion_uri, X_test, y_test
        )

        candidate_rmse = self.evaluate_model(
            self.candidate_uri, X_test, y_test
        )

        if not np.isfinite(
            [champion_rmse, candidate_rmse]
        ).all():
            raise ValueError("Model evaluation returned invalid RMSE.")

        if champion_rmse == 0:
            improvement = (
                0.0 if candidate_rmse == 0 else float("-inf")
            )
        else:
            improvement = (
                (champion_rmse - candidate_rmse)
                / champion_rmse
            )

        approved = improvement >= self.min_improvement

        return {
            "champion_rmse": champion_rmse,
            "candidate_rmse": candidate_rmse,
            "improvement": improvement,
            "approved": approved,
        }
