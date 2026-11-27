# House Pricing MLOps — Databricks & MLflow

An end-to-end **MLOps portfolio project** for predicting house prices with scikit-learn's linear regression, built around **Databricks**, **MLflow**, **Unity Catalog**, **Databricks Asset Bundles**, and **GitHub Actions**.

The project began with a single predictor (`square_footage`) and was extended to **multiple linear regression using seven numerical features**. Feature selection is driven primarily by YAML configuration, so new experiments can reuse the same training, evaluation, and model-lifecycle components.

> **Scope:** This is a hands-on learning/portfolio project, not a production-ready home valuation service. Training and validation are automated; model promotion and endpoint deployment remain explicitly controlled.

## Architecture

```text
Housing CSV
    |
    v
DataProcessor ----------> Unity Catalog Delta train/test tables
    |
    v
RegressionModel --------> MLflow experiment + logged model
    |
    v
ModelValidator ----------> Compare Candidate vs. Champion
    |                       (same test rows, each model's own features)
    |
    +-- Validation fails --> Keep current Champion; no registration
    |
    +-- Validation passes -> ModelLifecycle registers Candidate
                                      |
                                      v
                             Manual review / approval
                                      |
                                      v
                              ModelRelease
                             /            \
                     Promotion          Rollback
                         |                 |
                         v                 v
                      ModelDeployment --> Databricks serving endpoint
```

The Databricks training Job runs **four sequential tasks**:

| Task | Responsibility |
| --- | --- |
| `process_data` | Read and prepare the dataset; write train/test Delta tables |
| `train_model` | Train a regression model and log it to MLflow |
| `validate_model` | Compare the new model with the current Champion |
| `register_candidate` | Register a model version only when validation passes |

The training Job **does not automatically promote or deploy** a model.

## Technologies

| Area | Tools |
| --- | --- |
| Machine learning | Python, pandas, scikit-learn, NumPy |
| Data platform | Databricks, Apache Spark, Delta Lake, Unity Catalog |
| Experiment tracking & registry | MLflow |
| Workflow orchestration | Databricks Jobs |
| Infrastructure configuration | Databricks Asset Bundles |
| CI/CD and testing | GitHub Actions, pytest |
| Dependency management | uv |
| Online inference | Databricks Model Serving |

## Repository structure

```text
house-pricing-mlops/
├── .github/
│   └── workflows/
│       └── deploy-dev.yml
├── config/
│   └── project_config.yml
├── data/
│   └── house_prices.csv
├── notebooks/
│   ├── job_process_data.py
│   ├── job_train_model.py
│   ├── job_validate_model.py
│   ├── job_register_candidate.py
│   └── ...                       # test notebooks
├── resources/
│   └── training_job.yml
├── src/
│   └── house_price_simple_analysis/
│       ├── __init__.py
│       ├── config.py
│       ├── dataProcessor.py
│       ├── regression_model.py
│       ├── model_validator.py
│       ├── model_lifecycle.py
│       ├── model_deployment.py
│       └── model_release.py
├── tests/
│   └── test_regression_model.py
├── databricks.yml
├── pyproject.toml
└── uv.lock
```

> The tree reflects the components developed for this project. Check notebook extensions and filenames against your branch before publishing.

## Model and configurable features

The target is `house_price`. The current model is a **multiple linear regression** using these seven numerical inputs:

1. `square_footage`
2. `num_bedrooms`
3. `num_bathrooms`
4. `year_built`
5. `lot_size`
6. `garage_size`
7. `neighborhood_quality`

Feature selection is configured in `config/project_config.yml`:

```yaml
data:
  raw_data_path: data/house_prices.csv
  num_features:
    - square_footage
    - num_bedrooms
    - num_bathrooms
    - year_built
    - lot_size
    - garage_size
    - neighborhood_quality
  cat_features: []
  unnecessary_features: []
  target: house_price
  test_size: 0.2
  random_state: 42

databricks:
  catalog_name: mlops_dev
  schema_name: house_pricing
  train_table: train_set
  test_table: test_set

model:
  fit_intercept: true
  registered_model_name: house_price_model

mlflow:
  experiment_name: /Shared/house-pricing-linear-regression
```

Change the catalog, schema, and experiment path to match your environment. This is an example of the current configuration, not a promise that the sample names exist in every workspace.

### Experimenting with different feature sets

For example, to test four predictors, change `num_features`:

```yaml
num_features:
  - square_footage
  - num_bedrooms
  - num_bathrooms
  - neighborhood_quality
```

The processing and training code reads the configured numerical features; `LinearRegression` supports one or many predictors. The validator can compare models with **different input schemas** by selecting the feature columns each fitted model expects.

**Important constraints:**

- The evaluation data must contain **all columns needed by both** the new Candidate and the existing Champion. If preprocessing drops a Champion feature, preserve it in a separate stable evaluation dataset or adjust the data preparation logic; changing YAML alone is not enough.
- Adding or removing columns in existing Unity Catalog Delta tables may require **explicit schema changes**. In this project's Databricks environment, automatic schema migration was restricted by table ACLs.
- The serving request must supply the features expected by the **currently deployed** model version. The prediction method accepts flexible keyword arguments, but does not infer missing features.
- Compare candidates using the same holdout records. A dedicated, versioned evaluation dataset is preferable to the overwritten development test table for more rigorous experiments.

## Requirements

- A Databricks workspace with access to **Unity Catalog**, **Jobs**, and **MLflow**.
- Permission to create/update Delta tables and register Unity Catalog models.
- Access to **Databricks Model Serving** for the optional serving/release stage; availability depends on workspace capabilities.
- Python, [uv](https://docs.astral.sh/uv/), and the [Databricks CLI](https://docs.databricks.com/dev-tools/cli/index.html).
- A supported Databricks authentication method. The example GitHub Actions workflow uses a **personal access token (PAT)**.

## Setup

### 1. Clone the repository and install dependencies

```bash
git clone https://github.com/BrayanMtz314/house_price_mlops.git
cd house-pricing-mlops
uv sync
```

Ensure `data/house_prices.csv` exists and update `config/project_config.yml` for your workspace.

### 2. Run unit tests

On Linux/macOS:

```bash
PYTHONPATH=src uv run --with pytest pytest -v
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
uv run --with pytest pytest -v
```

These tests check the local training logic with synthetic data; they do not require a live Spark session or validate production model quality.

### 3. Configure Databricks authentication

For PAT-based CLI authentication:

```bash
databricks configure --token
```

Follow the prompts for your workspace host and token. Never commit credentials to Git.

### 4. Validate and deploy the Asset Bundle

```bash
databricks bundle validate -t dev
databricks bundle deploy -t dev
```

This creates or updates the bundle-managed **Job definition**; it does not run training.

### 5. Initialize the first Champion (first run only)

The validation task expects a registered model version with the `Champion` alias. Before running the full four-task pipeline for the first time, train and register a baseline model and assign the alias using MLflow / Unity Catalog. The serving endpoint is configured separately. You can load the tables to unity catalog and register your first model by following the `02_test_data_processor` and `03_test_regression_model` files. To enable the endpoint you must use the Databrick UI.

### 6. Run the training pipeline

```bash
databricks bundle run -t dev house_price_training
```

Inspect the four task outputs in Databricks and the associated MLflow experiment. The training task passes the logged model URI to the validation task using Databricks task values.

## Managing Unity Catalog schema changes

The project initially used only `square_footage`. When expanding to seven predictors, the existing Delta tables had to be updated to accept additional columns.

If your tables have the original three-column schema, the following is an **example one-time migration** (adjust catalog/schema names and verify existing columns first):

```sql
ALTER TABLE mlops_dev.house_pricing.train_set
ADD COLUMNS (
  num_bedrooms BIGINT,
  num_bathrooms BIGINT,
  year_built BIGINT,
  lot_size DOUBLE,
  garage_size BIGINT,
  neighborhood_quality BIGINT
);

ALTER TABLE mlops_dev.house_pricing.test_set
ADD COLUMNS (
  num_bedrooms BIGINT,
  num_bathrooms BIGINT,
  year_built BIGINT,
  lot_size DOUBLE,
  garage_size BIGINT,
  neighborhood_quality BIGINT
);
```

Check existing schemas with `DESCRIBE TABLE` before running migrations. **Do not rerun these `ADD COLUMNS` statements if the columns already exist.** Removing features or changing data types may require a different migration strategy. In this project, schema evolution was not automatically permitted, so SQL migrations were used instead.

## GitHub Actions CI/CD

The workflow in `.github/workflows/deploy-dev.yml` is intentionally triggered **manually** (`workflow_dispatch`). It:

1. Checks out the code and runs unit tests.
2. Validates the Databricks Asset Bundle.
3. Deploys the development Job after tests pass.

Configure these GitHub repository settings under **Settings → Secrets and variables → Actions**:

| Type | Name | Purpose |
| --- | --- | --- |
| Repository secret | `DATABRICKS_HOST` | Databricks workspace URL |
| Repository secret | `DATABRICKS_TOKEN` | Databricks PAT |

Use **Actions → Deploy Dev → Run workflow** to start the workflow. A successful CI/CD run means the code and Job configuration were tested/deployed; **it does not mean the Databricks Job has run or a model has been promoted**.

## Model validation, registration, and promotion

The project follows a **Candidate–Champion** strategy:

- **Candidate:** A newly trained model being evaluated.
- **Champion:** The approved model version used as the production reference.
- **Validation:** Both models are scored on the same evaluation rows using RMSE; each receives the input columns it was trained with.
- **Acceptance rule:** The Candidate must improve RMSE by at least **1%** to qualify for registration.
- **Registration:** Passing models receive a new Unity Catalog model version and a `Candidate` alias. Validation evidence is recorded as model-version tags, including the evaluated Champion version and Delta evaluation table version.
- **Promotion:** A manually selected **exact version** is checked against its validation evidence before moving the `Champion` alias.

The project uses tag keys with underscores, such as `validation_status` and `validation_champion_version`, to comply with Unity Catalog's tag-key restrictions.

### Controlled deployment

After a Candidate passes validation and is registered, use the `ModelRelease` class in a Databricks notebook:

```python
from house_price_simple_analysis.model_release import ModelRelease

release = ModelRelease(
    config=config,  # Loaded from project_config.yml
    endpoint_name="house-price-endpoint",
)

candidate = release.lifecycle.get_model_by_alias("Candidate")

# Preview without modifying aliases or the endpoint
release.promote_and_deploy(version=candidate.version, apply=False)

# Execute only after reviewing and approving the release
# release.promote_and_deploy(version=candidate.version, apply=True)
```

Promotion and endpoint deployment are distinct actions. **Changing the `Champion` alias alone does not update an existing serving endpoint.** `ModelRelease` coordinates the two operations and verifies the configured served version.

### Prediction example

The updated `ModelDeployment.predict()` method accepts named numerical features:

```python
response = release.deployment.predict(
    square_footage=2000,
    num_bedrooms=3,
    num_bathrooms=2,
    year_built=2015,
    lot_size=3500,
    garage_size=2,
    neighborhood_quality=7,
)
print(response)
```

Equivalent request payload for a compatible Databricks serving endpoint:

```json
{
  "dataframe_records": [
    {
      "square_footage": 2000.0,
      "num_bedrooms": 3.0,
      "num_bathrooms": 2.0,
      "year_built": 2015.0,
      "lot_size": 3500.0,
      "garage_size": 2.0,
      "neighborhood_quality": 7.0
    }
  ]
}
```

The request schema depends on the version currently served; the original single-feature model accepts only its original input schema.

### Rollback

`ModelRelease` records the previous Champion version when a release is applied and supports a manual rollback:

```python
# Replace with the exact version you deployed
release.rollback(release_version="<DEPLOYED_VERSION>", apply=False)

# Execute only after reviewing the rollback target
# release.rollback(release_version="<DEPLOYED_VERSION>", apply=True)
```

Rollback is **best-effort**, not transactional. If an endpoint update or rollback fails, inspect both the Champion alias and serving endpoint state before retrying.

## Key lessons and limitations

- A model's feature schema affects **data tables, evaluation, registry signatures, and serving requests**, not just the training code.
- Repeated development runs overwrite the train/test Delta tables. For stronger reproducibility, use a dedicated stable holdout and retain its version and source data.
- Model-version tags improve traceability but are not an immutable audit trail or a substitute for approval controls.
- Registration can create a partial version if tagging fails; production workflows should make registration idempotent and handle retries.
- Promotion/deployment and rollback require stronger health checks, concurrency controls, and monitoring for production use.
- The current implementation supports **numerical features**. Categorical variables would require encoding and a compatible preprocessing/inference pipeline.

## Potential improvements

- Experiment tracking for named feature sets and model configurations.
- Stable evaluation datasets and automated data-quality checks.
- Serving endpoint monitoring, latency checks, and prediction drift monitoring.
- Stronger release approvals and durable deployment history.
- Additional models and hyperparameter experiments beyond linear regression.

## Project purpose

This project demonstrates the engineering work required to move from a basic machine learning notebook to a repeatable **MLOps workflow**: data preparation, model training, tracking, validation, selective registration, controlled release, online inference, and rollback.

Its primary goal is to practice **modular Python design, Databricks workflows, MLflow model lifecycle management, and CI/CD**, rather than to optimize a real-world house-price prediction product.

