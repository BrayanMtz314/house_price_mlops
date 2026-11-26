
from house_price_simple_analysis.model_lifecycle import ModelLifecycle
from house_price_simple_analysis.model_deployment import ModelDeployment


class ModelRelease:
    """Coordinate model promotion, deployment, and rollback."""

    def __init__(self, config, endpoint_name="house-price-endpoint"):
        self.lifecycle = ModelLifecycle(config)
        self.deployment = ModelDeployment(
            config=config,
            endpoint_name=endpoint_name,
        )

    def get_deployed_version(self):
        """Read the model version currently configured on the endpoint."""

        endpoint = self.deployment.get_endpoint()
        entities = endpoint.config.served_entities

        if not entities or len(entities) != 1:
            raise ValueError("Expected a single-model serving endpoint.")

        entity = entities[0]

        if entity.entity_name != self.lifecycle.model_name:
            raise ValueError("Endpoint serves an unexpected model.")

        return str(entity.entity_version)

    def promote_and_deploy(self, version, apply=False):
        """Promote a validated version and deploy it."""

        version = str(version)

        champion = self.lifecycle.get_model_by_alias("Champion")
        previous_version = str(champion.version)
        deployed_version = self.get_deployed_version()

        if deployed_version != previous_version:
            raise ValueError(
                "Endpoint and Champion are not synchronized."
            )

        # Check validation evidence before making changes.
        self.lifecycle.promote_version(
            version=version,
            apply=False,
        )

        print(f"Current Champion: {previous_version}")
        print(f"New version: {version}")

        if not apply:
            print("Dry run: no changes made.")
            return

        # Save the previous version for a future rollback.
        self.lifecycle.client.set_model_version_tag(
            name=self.lifecycle.model_name,
            version=version,
            key="release_previous_champion",
            value=previous_version,
        )

        # Move the Champion alias.
        self.lifecycle.promote_version(
            version=version,
            apply=True,
        )

        try:
            # Update the endpoint to serve the new Champion.
            self.deployment.update_to_champion(apply=True)

            if self.get_deployed_version() != version:
                raise RuntimeError(
                    "Endpoint version verification failed."
                )

        except Exception:
            print("Deployment failed. Attempting rollback.")

            self.lifecycle.client.set_registered_model_alias(
                name=self.lifecycle.model_name,
                alias="Champion",
                version=previous_version,
            )

            # Restore the endpoint too.
            self.deployment.update_to_champion(apply=True)

            raise RuntimeError(
                "Deployment failed; previous version restored."
            )

        print(f"Version {version} deployed successfully.")

    def rollback(self, release_version, apply=False):
        """Restore the version preceding a specific release."""

        release_version = str(release_version)

        released_model = self.lifecycle.client.get_model_version(
            name=self.lifecycle.model_name,
            version=release_version,
        )

        previous_version = released_model.tags.get(
            "release_previous_champion"
        )

        if not previous_version:
            raise ValueError(
                "No previous Champion recorded for this release."
            )

        champion = self.lifecycle.get_model_by_alias("Champion")

        if str(champion.version) != release_version:
            raise ValueError(
                "The release is no longer the current Champion."
            )

        if self.get_deployed_version() != release_version:
            raise ValueError(
                "Endpoint is not serving the expected release."
            )

        print(f"Current version: {release_version}")
        print(f"Rollback target: {previous_version}")

        if not apply:
            print("Dry run: no changes made.")
            return

        self.lifecycle.client.set_registered_model_alias(
            name=self.lifecycle.model_name,
            alias="Champion",
            version=previous_version,
        )

        self.deployment.update_to_champion(apply=True)

        if self.get_deployed_version() != previous_version:
            raise RuntimeError(
                "Rollback endpoint verification failed."
            )

        print(f"Rollback completed: version {previous_version}")
