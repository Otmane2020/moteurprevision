"""
Quick smoke tests for the prediction engine.
Run: python -m pytest tests/ -v
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest

from app.metrics import mape, mae, rmse, r2_score, calculate_all_metrics
from app.data_processor import DataProcessor
from app.forecaster import ForecastEngine


# ---------------------------------------------------------------------------
# Sample monthly data (3 years)
# ---------------------------------------------------------------------------

def make_sample_df(n: int = 36, freq: str = "M") -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=n, freq=freq)
    np.random.seed(42)
    values = (
        100
        + np.arange(n) * 0.5
        + 10 * np.sin(2 * np.pi * np.arange(n) / 12)
        + np.random.normal(0, 3, n)
    )
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "value": values})


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_mape_perfect():
    a = np.array([100.0, 200.0, 300.0])
    assert mape(a, a) == 0.0


def test_mape_basic():
    a = np.array([100.0])
    p = np.array([110.0])
    assert abs(mape(a, p) - 10.0) < 1e-6


def test_all_metrics_keys():
    a = np.array([100.0, 90.0, 80.0])
    p = np.array([105.0, 88.0, 82.0])
    m = calculate_all_metrics(a, p)
    for key in ("mape", "smape", "mae", "rmse", "r2", "wape", "bias"):
        assert key in m


# ---------------------------------------------------------------------------
# DataProcessor
# ---------------------------------------------------------------------------


def test_detect_frequency_monthly():
    df = make_sample_df(24, "M")
    proc = DataProcessor()
    series, freq = proc.prepare_timeseries(df, "date", "value")
    assert "M" in freq or "MS" in freq


def test_train_test_split():
    df = make_sample_df(36, "M")
    proc = DataProcessor()
    series, _ = proc.prepare_timeseries(df)
    train, test = proc.train_test_split(series, 0.8)
    assert len(train) + len(test) == len(series)
    assert abs(len(train) / len(series) - 0.8) < 0.05


# ---------------------------------------------------------------------------
# Statistical models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name",
    ["naive_seasonal", "ses", "holtwinters", "arima", "theta"],
)
def test_statistical_model(model_name):
    from app.forecaster import ForecastEngine

    eng = ForecastEngine()
    df = make_sample_df(36)
    result = eng.run_forecast(df, models=[model_name], horizon=6)
    assert result["status"] == "success"
    assert model_name in result["forecasts"]
    fc = result["forecasts"][model_name]
    assert len(fc) == 6
    assert all(r["value"] >= 0 for r in fc)


# ---------------------------------------------------------------------------
# ML models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name", ["linear_regression", "random_forest", "xgboost", "lightgbm"]
)
def test_ml_model(model_name):
    eng = ForecastEngine()
    df = make_sample_df(48)  # more data for ML models
    result = eng.run_forecast(df, models=[model_name], horizon=6)
    assert result["status"] == "success"
    fc = result["forecasts"].get(model_name)
    assert fc is not None
    if isinstance(fc, list):
        assert len(fc) == 6


# ---------------------------------------------------------------------------
# Full engine
# ---------------------------------------------------------------------------


def test_full_engine_returns_best_model():
    eng = ForecastEngine()
    df = make_sample_df(48)
    result = eng.run_forecast(df, horizon=12)
    assert result["status"] == "success"
    assert result["best_model"] in eng.available_models()
    assert len(result["backtesting"]["rankings"]) > 0


def test_backtest_rankings_sorted_by_mape():
    eng = ForecastEngine()
    df = make_sample_df(48)
    bt = eng.run_backtest(df)
    rankings = bt["rankings"]
    mapes = [r["mape"] for r in rankings]
    assert mapes == sorted(mapes)


def test_multiproduct():
    df = make_sample_df(36)
    df["product"] = "P1"
    df2 = make_sample_df(36)
    df2["product"] = "P2"
    combined = pd.concat([df, df2], ignore_index=True)
    eng = ForecastEngine()
    result = eng.run_forecast(combined, product_column="product", horizon=6, models=["ses", "holtwinters"])
    assert "products" in result
    assert "P1" in result["products"]
    assert "P2" in result["products"]
