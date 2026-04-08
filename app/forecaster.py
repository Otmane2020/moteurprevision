"""
ForecastEngine – orchestrates all models, backtesting (80/20), and ranking.
"""
import warnings
import traceback
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from app.data_processor import DataProcessor
from app.metrics import calculate_all_metrics
from app.models.statistical import (
    NaiveForecaster,
    SESForecaster,
    HoltWintersForecaster,
    ARIMAForecaster,
    SARIMAForecaster,
    ThetaForecaster,
)
from app.models.ml_models import (
    LinearRegressionForecaster,
    RandomForestForecaster,
    XGBoostForecaster,
    LightGBMForecaster,
)
from app.models.prophet_model import ProphetForecaster, PROPHET_AVAILABLE

warnings.filterwarnings("ignore")

# Models that need seasonal_period kwarg
_SEASONAL_MODELS = {"naive_seasonal", "holtwinters", "sarima", "theta", "prophet"}


class ForecastEngine:
    """Main forecasting engine with 10 models + optional Prophet."""

    def __init__(self):
        self._processor = DataProcessor()
        self._registry: Dict[str, Any] = {
            "naive_seasonal": NaiveForecaster,
            "ses": SESForecaster,
            "holtwinters": HoltWintersForecaster,
            "arima": ARIMAForecaster,
            "sarima": SARIMAForecaster,
            "theta": ThetaForecaster,
            "linear_regression": LinearRegressionForecaster,
            "random_forest": RandomForestForecaster,
            "xgboost": XGBoostForecaster,
            "lightgbm": LightGBMForecaster,
        }
        if PROPHET_AVAILABLE:
            self._registry["prophet"] = ProphetForecaster

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def available_models(self) -> List[str]:
        return list(self._registry.keys())

    def model_info(self) -> List[Dict]:
        return [
            {
                "name": name,
                "description": getattr(cls, "description", name),
                "type": "statistical" if name in _SEASONAL_MODELS else "ml",
            }
            for name, cls in self._registry.items()
        ]

    # ------------------------------------------------------------------
    # Backtesting (80 / 20 split)
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        df: pd.DataFrame,
        date_column: str = "date",
        value_column: str = "value",
        models: List[str] = None,
        split_ratio: float = 0.8,
        frequency: Optional[str] = None,
    ) -> Dict:
        series, freq = self._processor.prepare_timeseries(df, date_column, value_column, frequency)
        train, test = self._processor.train_test_split(series, split_ratio)
        sp = self._processor.get_seasonal_period(freq)
        model_names = self._resolve_models(models)

        results: Dict[str, Any] = {}
        for name in model_names:
            results[name] = self._run_single_model(name, train, len(test), sp)
            if results[name].get("predictions") is not None:
                actual = test.values[: len(results[name]["predictions"])]
                pred = np.array(results[name]["predictions"])
                results[name]["metrics"] = calculate_all_metrics(actual, pred)
                results[name]["actual"] = actual.tolist()
                results[name]["dates"] = [str(d.date()) for d in test.index[: len(pred)]]

        rankings = self._rank(results)
        return {
            "status": "success",
            "frequency": freq,
            "seasonal_period": sp,
            "data_points": len(series),
            "train_size": len(train),
            "test_size": len(test),
            "split_ratio": split_ratio,
            "rankings": rankings,
            "details": {
                k: {
                    "metrics": v.get("metrics"),
                    "error": v.get("error"),
                }
                for k, v in results.items()
            },
        }

    # ------------------------------------------------------------------
    # Full forecast (backtest + future projection)
    # ------------------------------------------------------------------

    def run_forecast(
        self,
        df: pd.DataFrame,
        date_column: str = "date",
        value_column: str = "value",
        horizon: int = 12,
        frequency: Optional[str] = None,
        models: List[str] = None,
        product_column: Optional[str] = None,
    ) -> Dict:
        if product_column and product_column in df.columns:
            products = df[product_column].unique()
            out: Dict[str, Any] = {}
            for prod in products:
                sub = df[df[product_column] == prod]
                try:
                    out[str(prod)] = self._forecast_single(
                        sub, date_column, value_column, horizon, frequency, models
                    )
                except Exception as exc:
                    out[str(prod)] = {"error": str(exc)}
            return {"status": "success", "products": out}

        return self._forecast_single(df, date_column, value_column, horizon, frequency, models)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forecast_single(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_column: str,
        horizon: int,
        frequency: Optional[str],
        models: Optional[List[str]],
    ) -> Dict:
        series, freq = self._processor.prepare_timeseries(df, date_column, value_column, frequency)
        sp = self._processor.get_seasonal_period(freq)
        model_names = self._resolve_models(models)

        # Backtest first (80/20)
        backtest = self.run_backtest(df, date_column, value_column, model_names, 0.8, freq)

        # Future dates
        try:
            offset = pd.tseries.frequencies.to_offset(freq)
            future_idx = pd.date_range(start=series.index[-1] + offset, periods=horizon, freq=freq)
        except Exception:
            future_idx = pd.date_range(start=series.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="ME")

        forecasts: Dict[str, Any] = {}
        for name in model_names:
            res = self._run_single_model(name, series, horizon, sp)
            if res.get("predictions") is not None:
                forecasts[name] = [
                    {"date": str(d.date()), "value": round(float(v), 4), "model": name}
                    for d, v in zip(future_idx, res["predictions"])
                ]
            else:
                forecasts[name] = {"error": res.get("error", "unknown")}

        best = backtest["rankings"][0]["model"] if backtest["rankings"] else model_names[0]

        return {
            "status": "success",
            "best_model": best,
            "frequency": freq,
            "seasonal_period": sp,
            "horizon": horizon,
            "data_points": len(series),
            "date_range": {
                "start": str(series.index[0].date()),
                "end": str(series.index[-1].date()),
            },
            "backtesting": {
                "train_size": backtest["train_size"],
                "test_size": backtest["test_size"],
                "split_ratio": 0.8,
                "rankings": backtest["rankings"],
            },
            "forecasts": forecasts,
            "best_forecast": forecasts.get(best),
        }

    def _run_single_model(
        self, name: str, series: pd.Series, horizon: int, sp: int
    ) -> Dict:
        try:
            cls = self._registry[name]
            model = cls(seasonal_period=sp) if name in _SEASONAL_MODELS else cls()
            model.fit(series)
            preds = model.predict(horizon)
            return {"predictions": np.maximum(preds, 0).tolist()}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _resolve_models(self, models: Optional[List[str]]) -> List[str]:
        if not models or "all" in models:
            return list(self._registry.keys())
        return [m for m in models if m in self._registry]

    @staticmethod
    def _rank(results: Dict) -> List[Dict]:
        rows = []
        for name, res in results.items():
            m = res.get("metrics")
            if m and m.get("mape") is not None and m["mape"] != float("inf"):
                rows.append({"model": name, **m})
        rows.sort(key=lambda r: r.get("mape", float("inf")))
        for i, r in enumerate(rows):
            r["rank"] = i + 1
        return rows
