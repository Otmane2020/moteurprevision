"""
Statistical forecasting models:
  1. Naive Seasonal
  2. Simple Exponential Smoothing (SES)
  3. Holt-Winters Triple Exponential Smoothing
  4. ARIMA
  5. SARIMA
  6. Theta Method
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Seasonal Naive
# ---------------------------------------------------------------------------
class NaiveForecaster:
    name = "naive_seasonal"
    description = "Seasonal Naive – repeats last seasonal cycle (baseline)"

    def __init__(self, seasonal_period: int = 12):
        self.seasonal_period = seasonal_period
        self._last_season: np.ndarray = None

    def fit(self, series: pd.Series) -> "NaiveForecaster":
        sp = min(self.seasonal_period, len(series))
        self._last_season = series.values[-sp:]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        sp = len(self._last_season)
        return np.array([self._last_season[i % sp] for i in range(horizon)])


# ---------------------------------------------------------------------------
# 2. Simple Exponential Smoothing
# ---------------------------------------------------------------------------
class SESForecaster:
    name = "ses"
    description = "Simple Exponential Smoothing"

    def __init__(self, **_):
        self._result = None

    def fit(self, series: pd.Series) -> "SESForecaster":
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing

        model = SimpleExpSmoothing(series, initialization_method="estimated")
        self._result = model.fit(optimized=True, remove_bias=True)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.maximum(np.array(self._result.forecast(horizon)), 0)


# ---------------------------------------------------------------------------
# 3. Holt-Winters
# ---------------------------------------------------------------------------
class HoltWintersForecaster:
    name = "holtwinters"
    description = "Holt-Winters Triple Exponential Smoothing"

    def __init__(self, seasonal_period: int = 12):
        self.seasonal_period = seasonal_period
        self._result = None

    def fit(self, series: pd.Series) -> "HoltWintersForecaster":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        sp = max(2, min(self.seasonal_period, len(series) // 2))
        combos = [
            ("add", "add"),
            ("add", "mul"),
            ("mul", "add"),
            ("add", None),
        ]
        for trend, seasonal in combos:
            try:
                kwargs = dict(
                    trend=trend,
                    initialization_method="estimated",
                )
                if seasonal:
                    kwargs["seasonal"] = seasonal
                    kwargs["seasonal_periods"] = sp
                model = ExponentialSmoothing(series, **kwargs)
                self._result = model.fit(optimized=True, remove_bias=True)
                break
            except Exception:
                continue

        if self._result is None:
            from statsmodels.tsa.holtwinters import SimpleExpSmoothing

            self._result = SimpleExpSmoothing(series, initialization_method="estimated").fit()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.maximum(np.array(self._result.forecast(horizon)), 0)


# ---------------------------------------------------------------------------
# 4. ARIMA (simple grid search on AIC)
# ---------------------------------------------------------------------------
class ARIMAForecaster:
    name = "arima"
    description = "ARIMA – auto order selection (grid search AIC)"

    def __init__(self, **_):
        self._result = None
        self.order = (1, 1, 1)

    def fit(self, series: pd.Series) -> "ARIMAForecaster":
        from statsmodels.tsa.arima.model import ARIMA

        best_aic = float("inf")
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        m = ARIMA(series, order=(p, d, q))
                        r = m.fit()
                        if r.aic < best_aic:
                            best_aic = r.aic
                            self._result = r
                            self.order = (p, d, q)
                    except Exception:
                        continue

        if self._result is None:
            self._result = ARIMA(series, order=(1, 1, 1)).fit()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.array(self._result.forecast(steps=horizon))


# ---------------------------------------------------------------------------
# 5. SARIMA
# ---------------------------------------------------------------------------
class SARIMAForecaster:
    name = "sarima"
    description = "SARIMA – Seasonal ARIMA with automatic order selection"

    def __init__(self, seasonal_period: int = 12):
        self.seasonal_period = seasonal_period
        self._result = None

    def fit(self, series: pd.Series) -> "SARIMAForecaster":
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.arima.model import ARIMA

        sp = max(2, min(self.seasonal_period, len(series) // 3))
        candidates = [
            ((1, 1, 1), (1, 1, 0, sp)),
            ((1, 1, 1), (0, 1, 1, sp)),
            ((1, 1, 0), (1, 1, 0, sp)),
            ((0, 1, 1), (0, 1, 1, sp)),
            ((1, 1, 1), (0, 0, 0, 0)),
        ]
        best_aic = float("inf")
        for order, s_order in candidates:
            try:
                m = SARIMAX(series, order=order, seasonal_order=s_order)
                r = m.fit(disp=False, maxiter=100)
                if r.aic < best_aic:
                    best_aic = r.aic
                    self._result = r
            except Exception:
                continue

        if self._result is None:
            self._result = ARIMA(series, order=(1, 1, 1)).fit()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.array(self._result.forecast(steps=horizon))


# ---------------------------------------------------------------------------
# 6. Theta Method (M3 competition winner)
# ---------------------------------------------------------------------------
class ThetaForecaster:
    name = "theta"
    description = "Theta Method – M3 forecasting competition winner"

    def __init__(self, seasonal_period: int = 12):
        self.seasonal_period = seasonal_period
        self._result = None
        self._fallback: HoltWintersForecaster = None

    def fit(self, series: pd.Series) -> "ThetaForecaster":
        try:
            from statsmodels.tsa.forecasting.theta import ThetaModel

            period = max(2, min(self.seasonal_period, len(series) // 2))
            m = ThetaModel(series, period=period)
            self._result = m.fit()
        except Exception:
            self._fallback = HoltWintersForecaster(self.seasonal_period)
            self._fallback.fit(series)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._result is not None:
            return np.maximum(np.array(self._result.forecast(horizon)), 0)
        return self._fallback.predict(horizon)
