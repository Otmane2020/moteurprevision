"""
Machine Learning forecasting models (lag-feature based):
  7. Linear Regression
  8. Random Forest
  9. XGBoost
  10. LightGBM
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Feature engineering for ML models
# ---------------------------------------------------------------------------

def _create_features(series: pd.Series, n_lags: int = 12):
    """
    Build a feature matrix from a time series using:
    - Lag values (lag_1 … lag_n)
    - Rolling mean / std
    - Calendar features (month, quarter, year)
    - Linear trend
    Returns X (ndarray), y (ndarray), feature_names (list)
    """
    n = len(series)
    effective_lags = min(n_lags, max(1, n // 4))

    df = pd.DataFrame({"y": series.values}, index=series.index)

    for lag in range(1, effective_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)

    for w in [3, 6, 12]:
        if w < n // 3:
            df[f"roll_mean_{w}"] = df["y"].rolling(w, min_periods=1).mean().shift(1)
            df[f"roll_std_{w}"] = df["y"].rolling(w, min_periods=1).std().shift(1).fillna(0)

    if hasattr(series.index, "month"):
        df["month"] = series.index.month
        df["quarter"] = series.index.quarter
        df["year"] = series.index.year - series.index.year.min()

    df["trend"] = np.arange(n)
    df = df.dropna()

    feature_names = [c for c in df.columns if c != "y"]
    X = df[feature_names].values
    y = df["y"].values
    return X, y, feature_names


def _next_date(index):
    """Infer next date from index delta."""
    if len(index) >= 2:
        return index[-1] + (index[-1] - index[-2])
    return index[-1] + pd.DateOffset(months=1)


# ---------------------------------------------------------------------------
# Base ML Forecaster
# ---------------------------------------------------------------------------

class BaseMLForecaster:
    name = "ml_base"
    description = "Base ML model"

    def __init__(self, n_lags: int = 12):
        self.n_lags = n_lags
        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        self._series: pd.Series = None
        self._n_features: int = 0
        self._feature_names: list = []
        self.model = None

    # ------------------------------------------------------------------
    def fit(self, series: pd.Series) -> "BaseMLForecaster":
        self._series = series.copy()
        X, y, self._feature_names = _create_features(series, self.n_lags)

        if len(X) < 5:
            raise ValueError("Not enough data for ML model (need >= 5 non-NaN rows after lagging)")

        self._n_features = X.shape[1]
        Xs = self._scaler_X.fit_transform(X)
        ys = self._scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        self._fit_model(Xs, ys)
        return self

    def _fit_model(self, X, y):
        raise NotImplementedError

    # ------------------------------------------------------------------
    def predict(self, horizon: int) -> np.ndarray:
        """Recursive multi-step forecast."""
        ext_values = list(self._series.values)
        ext_index = list(self._series.index)
        preds = []

        for _ in range(horizon):
            tmp = pd.Series(ext_values, index=pd.DatetimeIndex(ext_index))
            X_all, _, _ = _create_features(tmp, self.n_lags)

            if len(X_all) == 0:
                preds.append(float(ext_values[-1]))
            else:
                x_last = X_all[-1].reshape(1, -1)
                # Pad/truncate to match training feature count
                n = self._n_features
                if x_last.shape[1] < n:
                    x_last = np.pad(x_last, ((0, 0), (0, n - x_last.shape[1])))
                elif x_last.shape[1] > n:
                    x_last = x_last[:, :n]

                xs = self._scaler_X.transform(x_last)
                ys = self.model.predict(xs)
                y_hat = float(
                    self._scaler_y.inverse_transform(ys.reshape(-1, 1))[0][0]
                )
                y_hat = max(0.0, y_hat)
                preds.append(y_hat)

            next_dt = _next_date(ext_index)
            ext_values.append(preds[-1])
            ext_index.append(next_dt)

        return np.array(preds)


# ---------------------------------------------------------------------------
# Concrete models
# ---------------------------------------------------------------------------

class LinearRegressionForecaster(BaseMLForecaster):
    name = "linear_regression"
    description = "Linear Regression with lag & calendar features"

    def _fit_model(self, X, y):
        self.model = LinearRegression()
        self.model.fit(X, y)


class RandomForestForecaster(BaseMLForecaster):
    name = "random_forest"
    description = "Random Forest Regressor"

    def _fit_model(self, X, y):
        self.model = RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            min_samples_split=3,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y)


class XGBoostForecaster(BaseMLForecaster):
    name = "xgboost"
    description = "XGBoost Gradient Boosting"

    def _fit_model(self, X, y):
        import xgboost as xgb

        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )
        self.model.fit(X, y)


class LightGBMForecaster(BaseMLForecaster):
    name = "lightgbm"
    description = "LightGBM Gradient Boosting"

    def _fit_model(self, X, y):
        import lightgbm as lgb

        self.model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        self.model.fit(X, y)
