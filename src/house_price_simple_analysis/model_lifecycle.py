
import mlflow
from mlflow import MlflowClient

from house_price_simple_analysis.config import ProjectConfig


class ModelLifecycle:
    """Manage registered model versions and aliases."""

    def __init__(self, config: ProjectConfig):
        self.config = config

        self.model_name = (
            f"{config.databricks.catalog_name}."
            f"{config.databricks.schema_name}."
            f"{config.model.registered_model_name}"
        )

        mlflow.set_registry_uri("databricks-uc")
        self.client = MlflowClient(registry_uri="databricks-uc")

    def register_candidate(self, model_uri: str) -> str:
        """Register a logged model and assign the Candidate alias."""

        registered = mlflow.register_model(
            model_uri=model_uri,
            name=self.model_name,
        )

        self.client.set_registered_model_alias(
            name=self.model_name,
            alias="Candidate",
            version=registered.version,
        )

        return f"models:/{self.model_name}/{registered.version}"

    def get_model_by_alias(self, alias: str):
        """Retrieve metadata for a model version by alias."""

        return self.client.get_model_version_by_alias(
            name=self.model_name,
            alias=alias,
        )

    def promote_candidate(
        self,
        approved: bool,
        apply: bool = False,
    ) -> None:
        """Promote Candidate to Champion after validation."""

        if not approved:
            print("Candidate rejected. Champion remains unchanged.")
            return

        candidate = self.get_model_by_alias("Candidate")

        print(f"Candidate version: {candidate.version}")

        if not apply:
            print("Dry run: no changes made.")
            return

        self.client.set_registered_model_alias(
            name=self.model_name,
            alias="Champion",
            version=candidate.version,
        )

        print(f"Version {candidate.version} promoted to Champion.")
