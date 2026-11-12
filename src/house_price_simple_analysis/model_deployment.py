
import mlflow

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput
from mlflow import MlflowClient

from house_price_simple_analysis.config import ProjectConfig


class ModelDeployment:
    """Manage an existing Databricks Model Serving endpoint."""

    def __init__(
        self,
        config: ProjectConfig,
        endpoint_name: str,
        alias: str = "Champion",
    ):
        self.config = config
        self.endpoint_name = endpoint_name
        self.alias = alias

        self.model_name = (
            f"{config.databricks.catalog_name}."
            f"{config.databricks.schema_name}."
            f"{config.model.registered_model_name}"
        )

        mlflow.set_registry_uri("databricks-uc")

        self.mlflow_client = MlflowClient()
        self.workspace = WorkspaceClient()

    def get_champion_version(self) -> str:
        """Resolve the model version assigned to the alias."""

        model_version = (
            self.mlflow_client.get_model_version_by_alias(
                name=self.model_name,
                alias=self.alias,
            )
        )

        return str(model_version.version)

    def get_endpoint(self):
        """Retrieve the existing serving endpoint."""

        return self.workspace.serving_endpoints.get(
            name=self.endpoint_name
        )

    def predict(self, square_footage: float):
        """Request a prediction from the serving endpoint."""

        response = self.workspace.serving_endpoints.query(
            name=self.endpoint_name,
            dataframe_records=[
                {"square_footage": float(square_footage)}
            ],
        )

        return response.predictions

    def update_to_champion(self, apply: bool = False):
        """Update a single-model endpoint to the Champion version."""

        champion_version = self.get_champion_version()
        endpoint = self.get_endpoint()

        entities = endpoint.config.served_entities or []

        if len(entities) != 1:
            raise ValueError(
                "This method supports only single-model endpoints."
            )

        current = entities[0]

        if current.entity_name != self.model_name:
            raise ValueError(
                "Endpoint is serving a different registered model."
            )

        current_version = str(current.entity_version)

        if current_version == champion_version:
            print("Endpoint already serves the Champion version.")
            return endpoint

        print(f"Current version: {current_version}")
        print(f"Champion version: {champion_version}")

        if not apply:
            print("Dry run: no endpoint changes were made.")
            return None

        return self.workspace.serving_endpoints.update_config_and_wait(
            name=self.endpoint_name,
            served_entities=[
                ServedEntityInput(
                    name=current.name,
                    entity_name=self.model_name,
                    entity_version=champion_version,
                    workload_size=current.workload_size,
                    workload_type=current.workload_type,
                    scale_to_zero_enabled=current.scale_to_zero_enabled,
                )
            ],
        )
