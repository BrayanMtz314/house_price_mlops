
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

    def register_candidate(
        self,
        model_uri: str,
        evidence: dict,
    ) -> str:
        """Register a validated candidate with its evaluation evidence."""

        if evidence.get("approved") is not True:
            raise ValueError("Candidate did not pass validation.")

        if evidence.get("candidate_uri") != model_uri:
            raise ValueError("Validation belongs to a different model.")

        required = [
            "champion_version",
            "evaluation_table",
            "evaluation_version",
            "candidate_rmse",
            "champion_rmse",
            "improvement",
            "minimum_improvement",
        ]

        for field in required:
            if field not in evidence:
                raise ValueError(f"Missing validation evidence: {field}")

        registered = mlflow.register_model(
            model_uri=model_uri,
            name=self.model_name,
        )

        version = str(registered.version)

        tags = {
            "validation.status": "passed",
            "validation.model_uri": model_uri,
            "validation.champion_version": evidence["champion_version"],
            "validation.evaluation_table": evidence["evaluation_table"],
            "validation.evaluation_version": evidence["evaluation_version"],
            "validation.candidate_rmse": evidence["candidate_rmse"],
            "validation.champion_rmse": evidence["champion_rmse"],
            "validation.improvement": evidence["improvement"],
            "validation.minimum_improvement": evidence["minimum_improvement"],
        }

        for key, value in tags.items():
            self.client.set_model_version_tag(
                name=self.model_name,
                version=version,
                key=key,
                value=str(value),
            )

        self.client.set_registered_model_alias(
            name=self.model_name,
            alias="Candidate",
            version=version,
        )

        return f"models:/{self.model_name}/{version}"

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
        
    
    def promote_version(
        self,
        version: str,
        apply: bool = False,
    ) -> None:
        """Promote an exact validated model version."""

        version = str(version)

        model_version = self.client.get_model_version(
            name=self.model_name,
            version=version,
        )

        tags = model_version.tags or {}

        if tags.get("validation.status") != "passed":
            raise ValueError(
                "This model version has no passing validation evidence."
            )

        champion = self.get_model_by_alias("Champion")

        evaluated_champion = tags.get(
            "validation.champion_version"
        )

        if evaluated_champion != str(champion.version):
            raise ValueError(
                "Champion has changed since validation. "
                "Re-evaluate the candidate before promotion."
            )

        improvement = float(tags["validation.improvement"])
        threshold = float(tags["validation.minimum_improvement"])

        if improvement < threshold:
            raise ValueError(
                "Candidate does not meet the improvement threshold."
            )

        print(f"Current Champion: version {champion.version}")
        print(f"Proposed Champion: version {version}")

        if not apply:
            print("Dry run: no changes made.")
            return

        self.client.set_registered_model_alias(
            name=self.model_name,
            alias="Champion",
            version=version,
        )

        print(f"Version {version} promoted to Champion.")

