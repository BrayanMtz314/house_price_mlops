
from pathlib import Path

import pandas as pd
import numpy as np

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

    rng = np.random.default_rng(42)

    features = config.data.num_features

    feature_data = {
        feature: rng.uniform(0, 1, 60)
        for feature in features
    }

    df = pd.DataFrame(feature_data)

    # Synthetic linear relationship
    df[config.data.target] = (
        200000
        + sum(
            (i + 1) * 5000 * df[feature]
            for i, feature in enumerate(features)
        )
    )

    regression.train_set = df.iloc[:45].copy()
    regression.test_set = df.iloc[45:].copy()
    

    # Execute model methods
    regression.prepare_model()
    regression.train()
    metrics = regression.evaluate()

    # Validate expected behavior
    assert metrics["r2"] > 0.99
    assert metrics["rmse"] < 1e-6
