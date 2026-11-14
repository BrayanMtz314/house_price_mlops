
from pathlib import Path

import pandas as pd

from house_price_simple_analysis.config import ProjectConfig
from house_price_simple_analysis.regression_model import RegressionModel


def test_regression_model_training():
    # Load project configuration
    root = Path(__file__).resolve().parents[1]
    config = ProjectConfig.from_yaml(
        str(root / "config" / "project_config.yml")
    )

    # Create a model without connecting to Databricks
    regression = RegressionModel(
        config=config,
        spark=None
    )

    # Create a small synthetic dataset
    regression.train_set = pd.DataFrame({
        "square_footage": [
            1000.0, 1500.0, 2000.0, 2500.0, 3000.0
        ],
        "house_price": [
            150000.0, 200000.0, 250000.0, 300000.0, 350000.0
        ]
    })

    regression.test_set = pd.DataFrame({
        "square_footage": [1200.0, 2200.0],
        "house_price": [170000.0, 270000.0]
    })

    # Execute model methods
    regression.prepare_model()
    regression.train()
    metrics = regression.evaluate()

    # Validate expected behavior
    assert metrics["r2"] > 0.99
    assert metrics["rmse"] < 1e-6
