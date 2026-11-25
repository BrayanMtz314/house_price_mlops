# House Pricing MLOps — Databricks & MLflow

An end-to-end **MLOps learning project** that predicts house prices with scikit-learn linear regression and demonstrates how to train, evaluate, track, register, and serve models using **Databricks**, **MLflow**, **Unity Catalog**, **Databricks Asset Bundles**, and **GitHub Actions**.

The initial model uses `square_footage` to predict `house_price`. The pipeline is configuration-driven so additional numeric features can be introduced later.

> **Project scope:** A portfolio/learning implementation, not a production-ready housing valuation system. Training and validation are automated; model promotion and endpoint deployment require explicit manual action.

## Architecture

```text
CSV dataset
    │
    ▼
DataProcessor ──► Unity Catalog Delta train/test tables
    │
    ▼
RegressionModel ──► MLflow experiment + logged model
    │
    ▼
ModelValidator ──► Compare candidate and Champion on test data
    │
    ├── Rejected ──► Keep current Champion
    │
    └── Passed ──► ModelLifecycle registers Candidate
                          │
                          ▼
                   Manual approval
                          │
                          ▼
                   ModelRelease / ModelDeployment
                          │
                          ▼
                   Databricks serving endpoint
```

The Databricks Job runs four sequential tasks: `process_data`, `train_model`, `validate_model`, and `register_candidate`. A rejected model remains logged in MLflow but is not registered by the automated Job.

## Tech stack

- **Python, pandas, scikit-learn** — data preparation and linear regression
- **Databricks, Spark, Delta Lake, Unity Catalog** — managed data and model assets
- **MLflow** — experiment tracking, model logging, registry versions, and aliases (`Candidate` / `Champion`)
- **Databricks Jobs & Asset Bundles** — pipeline orchestration and infrastructure deployment
- **GitHub Actions, pytest, uv** — manual CI/CD workflow, tests, and Python dependency management
- **Databricks Model Serving** — online inference

## Repository structure

```text
house-pricing-mlops/
├── .github/workflows/deploy-dev.yml
├── config/project_config.yml
├── data/house_prices.csv
├── notebooks/
│   ├── job_process_data.py
│   ├── job_train_model.py
│   ├── job_validate_model.py
│   └── job_register_candidate.py
├── resources/training_job.yml
├── src/house_price_simple_analysis/
│   ├── config.py
│   ├── dataProcessor.py
│   ├── regression_model.py
│   ├── model_validator.py
│   ├── model_lifecycle.py
│   ├── model_deployment.py
│   └── model_release.py
├── tests/test_regression_model.py
├── databricks.yml
├── pyproject.toml
└── uv.lock
```

> Notebook names/extensions and the presence of `model_release.py` may vary with your current branch; update this tree to match your repository.

## Prerequisites

1. A Databricks workspace with access to **Unity Catalog**, **Jobs**, and **MLflow**. Model Serving access is required for the deployment stage; availability can depend on workspace capabilities.
2. Python, [`uv`](https://docs.astral.sh/uv/), and the [Databricks CLI](https://docs.databricks.com/dev-tools/cli/index.html).
3. A Databricks catalog and schema with permissions to create/write Delta tables and register models.
4. A personal access token (PAT) or another supported Databricks authentication method. This example uses a PAT, as in the project's GitHub Actions workflow.

## Local setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/BrayanMtz314/house_price_mlops.git
cd house-pricing-mlops
uv sync
```

Review `config/project_config.yml` and change the catalog, schema, experiment name, and other settings to match your workspace:

```yaml
data:
  num_features: [square_footage]
  target: house_price

databricks:
  catalog_name: mlops_dev
  schema_name: house_pricing
  train_table: train_set
  test_table: test_set

model:
  registered_model_name: house_price_model
```

This is an **illustrative excerpt**, not a replacement for the complete configuration file. Ensure the CSV file exists at the configured `raw_data_path` and the workspace catalog/schema exist before running the Job.

Run unit tests:

```bash
PYTHONPATH=src uv run --with pytest pytest -v
```

On Windows PowerShell, set the import path first with `$env:PYTHONPATH = 'src'` and then run `uv run --with pytest pytest -v`.

## Databricks authentication and deployment

For local development, configure the Databricks CLI with your workspace URL and PAT:

```bash
databricks configure --token
```

Keep tokens out of source control. Then validate, deploy, and run the development bundle:

```bash
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run -t dev house_price_training
```

The bundle defines the four-task training pipeline. Inspect the Job run in Databricks to review task outputs and the MLflow experiment.

### First-time initialization

The validator compares each new model against the registered `Champion` alias. **Before the first automated validation run**, you must train/register an initial baseline model and assign it the `Champion` alias in Unity Catalog. Otherwise, there is no Champion to compare against.

For online inference, create/configure a compatible serving endpoint separately and point it at the registered Champion version. The endpoint is **not** automatically updated when a model alias changes.

## GitHub Actions deployment

The workflow `.github/workflows/deploy-dev.yml` is started manually with **Actions → Deploy Dev → Run workflow** (`workflow_dispatch`). It:

1. Checks out the repository and runs pytest.
2. Validates the Databricks Asset Bundle.
3. Deploys the development Job only if the tests pass.

Configure these repository settings under **Settings → Secrets and variables → Actions**:

| Type | Name | Value |
| --- | --- | --- |
| Variable | `DATABRICKS_HOST` | Workspace URL (`https://...`) |
| Secret | `DATABRICKS_TOKEN` | Databricks PAT |

A successful workflow **deploys the Job definition**; it does not automatically execute training or promote a model. Run the Databricks Job separately.

## Model validation and release

- **Validation:** Evaluate the logged Candidate and the current Champion against the same test rows, comparing RMSE. The sample threshold requires at least **1% RMSE improvement**.
- **Registration:** Only passing candidates are registered by the Job. Validation evidence is stored with the registered model version.
- **Promotion:** Review a specific registered version and its validation evidence before moving the `Champion` alias. Promotion is manual.
- **Deployment:** Update the serving endpoint to the approved model version and verify predictions. Moving an alias alone does not update the endpoint.
- **Rollback:** The release component is designed to restore the previous version, subject to endpoint update success. Rollback is not guaranteed to be atomic.

The initial model's prediction request contains one feature:

```json
{"dataframe_records": [{"square_footage": 2000.0}]}
```

## Limitations and next steps

- The sample data and deterministic linear regression may produce identical metrics across runs; rejecting a new model is expected in that case.
- Development train/test tables are overwritten on each run. A production system should use a stable, versioned holdout dataset and prevent train/test leakage.
- The registry tags and manual approval flow are educational safeguards, not a tamper-proof audit or fully transactional release mechanism.
- Adding numerical predictors is mostly configuration-driven, but the serving prediction interface must also be updated to accept the new feature schema.
- Future improvements: richer feature engineering, data quality checks, automated monitoring, release approvals, and more robust deployment/rollback checks.


