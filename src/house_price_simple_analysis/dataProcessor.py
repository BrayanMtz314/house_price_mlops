
"""Data processor class, responsible for transforming raw data
into datasets ready for model training."""

import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp
from sklearn.model_selection import train_test_split
from pathlib import Path

from house_price_simple_analysis.config import ProjectConfig


class DataProcessor:

    def __init__(self, config: ProjectConfig, spark: SparkSession, project_root: Path):
        self.config = config
        self.spark = spark
        self.project_root = project_root

        self.data_path = self.project_root / config.data.raw_data_path
        self.catalog_name = config.databricks.catalog_name
        self.schema_name = config.databricks.schema_name

        self.df = None
        self.train_set = None
        self.test_set = None

    def load_data(self) -> pd.DataFrame:
        """Load raw data from the specified path."""
        self.df = pd.read_csv(self.data_path)
        return self.df

    def preprocess(self) -> pd.DataFrame:
        """Clean and validate the dataset."""

        if self.df is None:
            raise ValueError("Load the data before preprocessing.")

        # Normalize column names
        self.df.columns = [col.lower() for col in self.df.columns]

        # Validate required columns
        required_columns = (
            self.config.data.num_features
            + self.config.data.cat_features
            + [self.config.data.target]
        )

        missing_columns = [
            col for col in required_columns if col not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Drop unnecessary features
        self.df = self.df.drop(
            columns=self.config.data.unnecessary_features,
            errors="ignore"
        )

        # Handle missing values
        self.df = self.df.dropna()

        if self.df.empty:
            raise ValueError("No rows remaining after preprocessing.")

        return self.df

    def split_data(self):
        """Split the dataset into training and testing sets."""

        if self.df is None:
            raise ValueError("Load and preprocess the data before splitting.")

        features = (
            self.config.data.num_features
            + self.config.data.cat_features
        )

        X = self.df[features]
        y = self.df[self.config.data.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.data.test_size,
            random_state=self.config.data.random_state
        )

        self.train_set = (X_train, y_train)
        self.test_set = (X_test, y_test)

        return self.train_set, self.test_set

    def save_to_catalog(self) -> None:
        """Save the training and testing datasets to Unity Catalog."""

        if self.train_set is None or self.test_set is None:
            raise ValueError("Split the data before saving.")

        X_train, y_train = self.train_set
        X_test, y_test = self.test_set

        train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
        test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

        # Convert to Spark DataFrames and add timestamps
        train_spark_df = self.spark.createDataFrame(train_df).withColumn(
            "update_timestamp_utc", current_timestamp()
        )

        test_spark_df = self.spark.createDataFrame(test_df).withColumn(
            "update_timestamp_utc", current_timestamp()
        )

        # Save to Unity Catalog
        train_table = (
            f"{self.catalog_name}.{self.schema_name}."
            f"{self.config.databricks.train_table}"
        )

        test_table = (
            f"{self.catalog_name}.{self.schema_name}."
            f"{self.config.databricks.test_table}"
        )

        train_spark_df.write.mode("overwrite").saveAsTable(train_table)
        test_spark_df.write.mode("overwrite").saveAsTable(test_table)
