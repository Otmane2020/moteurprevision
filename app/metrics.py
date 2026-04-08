"""
KPI Metrics for forecast evaluation.
"""
import numpy as np
from typing import Dict


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    mask = actual != 0
    if not mask.any():
        return float("inf")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric MAPE (%)."""
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    mask = denom != 0
    if not mask.any():
        return float("inf")
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.array(actual, dtype=float) - np.array(predicted, dtype=float))))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Square Error."""
    return float(
        np.sqrt(np.mean((np.array(actual, dtype=float) - np.array(predicted, dtype=float)) ** 2))
    )


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """R² coefficient of determination."""
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (%)."""
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    total = np.sum(np.abs(actual))
    if total == 0:
        return float("inf")
    return float(np.sum(np.abs(actual - predicted)) / total * 100)


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Forecast Bias."""
    return float(np.mean(np.array(predicted, dtype=float) - np.array(actual, dtype=float)))


def calculate_all_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Calculate all KPI metrics."""
    return {
        "mape": round(mape(actual, predicted), 4),
        "smape": round(smape(actual, predicted), 4),
        "mae": round(mae(actual, predicted), 4),
        "rmse": round(rmse(actual, predicted), 4),
        "r2": round(r2_score(actual, predicted), 4),
        "wape": round(wape(actual, predicted), 4),
        "bias": round(bias(actual, predicted), 4),
    }
