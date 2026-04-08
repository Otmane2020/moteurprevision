"""
Facebook Prophet model (optional – graceful fallback if not installed).
Model 11 – used when available; always included in rankings if installed.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROPHET_AVAILABLE = False
try:
    from prophet import Prophet  # noqa: F401

    PROPHET_AVAILABLE = True
except Exception:
    pass


class ProphetForecaster:
    name = "prophet"
    description = "Facebook Prophet – trend + seasonality + holidays"
    available = PROPHET_AVAILABLE

    def __init__(self, seasonal_period: int = 12):
        self._model = None
        self._freq: str = "M"

    def fit(self, series: pd.Series) -> "ProphetForecaster":
        if not PROPHET_AVAILABLE:
            raise ImportError("prophet not installed")

        from prophet import Prophet

        freq = pd.infer_freq(series.index) or "M"
        self._freq = freq

        df = pd.DataFrame({"ds": series.index, "y": series.values}).reset_index(drop=True)

        self._model = Prophet(
            yearly_seasonality="auto",
            weekly_seasonality="auto",
            daily_seasonality=False,
            interval_width=0.95,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(df)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        future = self._model.make_future_dataframe(periods=horizon, freq=self._freq)
        forecast = self._model.predict(future)
        return np.maximum(np.array(forecast["yhat"].tail(horizon)), 0)
