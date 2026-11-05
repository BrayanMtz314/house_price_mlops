import yaml
from pydantic import BaseModel


class ProjectSettings(BaseModel):
    name: str
    environment: str = "dev"


class DataSettings(BaseModel):
    raw_data_path: str

    num_features: list[str]
    cat_features: list[str] = []
    
    unnecessary_features: list[str]

    target: str

    test_size: float = 0.2
    random_state: int = 42


class DatabricksSettings(BaseModel):
    catalog_name: str
    schema_name: str

    train_table: str
    test_table: str


class ModelSettings(BaseModel):
    fit_intercept: bool = True


class MLflowSettings(BaseModel):
    experiment_name: str


class ProjectConfig(BaseModel):
    project: ProjectSettings
    data: DataSettings
    databricks: DatabricksSettings
    model: ModelSettings
    mlflow: MLflowSettings

    @classmethod
    def from_yaml(cls, config_path: str) -> "ProjectConfig":

        with open(config_path) as file:
            config_dict = yaml.safe_load(file)

        return cls(**config_dict)