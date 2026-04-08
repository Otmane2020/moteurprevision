"""
Data ingestion and preprocessing for time series forecasting.
Supports: CSV upload, JSON API, SAP/ERP format.
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import io


FREQ_MAP = {
    "D": 7,    # daily -> weekly seasonality
    "B": 5,    # business days
    "W": 52,   # weekly -> yearly seasonality
    "ME": 12,  # month-end (pandas >= 2.2)
    "MS": 12,  # month-start
    "M": 12,   # legacy alias
    "QE": 4,   # quarter-end
    "QS": 4,   # quarter-start
    "Q": 4,    # legacy
    "YE": 1,   # year-end
    "YS": 1,   # year-start
    "A": 1,    # legacy
    "Y": 1,    # legacy
    "H": 24,   # hourly -> daily seasonality
    "h": 24,
}

# Normalize old aliases to new pandas >= 2.2 style
_FREQ_NORMALIZE = {
    "M": "ME",
    "Q": "QE",
    "A": "YE",
    "Y": "YE",
}

FREQ_LABELS = {
    "D": "Daily",
    "W": "Weekly",
    "M": "Monthly",
    "Q": "Quarterly",
    "Y": "Annual",
    "H": "Hourly",
}


class DataProcessor:

    @staticmethod
    def prepare_timeseries(
        df: pd.DataFrame,
        date_column: str = "date",
        value_column: str = "value",
        frequency: Optional[str] = None,
    ) -> Tuple[pd.Series, str]:
        """
        Parse, clean and resample a DataFrame into a pd.Series.
        Returns (series, frequency_string).
        """
        if date_column not in df.columns:
            raise ValueError(f"Column '{date_column}' not found. Available: {list(df.columns)}")
        if value_column not in df.columns:
            raise ValueError(f"Column '{value_column}' not found. Available: {list(df.columns)}")

        df = df[[date_column, value_column]].copy()
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.dropna(subset=[date_column])
        df[value_column] = pd.to_numeric(df[value_column], errors="coerce").fillna(0)
        df = df.sort_values(date_column).drop_duplicates(subset=[date_column])

        series = df.set_index(date_column)[value_column].astype(float)

        # Detect or validate frequency
        detected = DataProcessor.detect_frequency(series)
        raw_freq = frequency or detected
        freq = _FREQ_NORMALIZE.get(raw_freq.upper(), raw_freq)

        # Resample to fill gaps
        try:
            series = series.resample(freq).sum()
        except Exception:
            pass

        # Fill missing values
        series = series.interpolate(method="linear").ffill().bfill().fillna(0)

        if len(series) < 4:
            raise ValueError(f"Not enough data points ({len(series)}). Need at least 4.")

        return series, freq

    @staticmethod
    def detect_frequency(series: pd.Series) -> str:
        """Auto-detect time series frequency from index."""
        try:
            inferred = pd.infer_freq(series.index)
            if inferred:
                return _FREQ_NORMALIZE.get(inferred.upper(), inferred)
        except Exception:
            pass

        if len(series) < 2:
            return "M"

        diffs = series.index.to_series().diff().dropna()
        median_days = diffs.median().days

        if median_days <= 1:
            return "D"
        elif median_days <= 7:
            return "W"
        elif median_days <= 35:
            return "ME"
        elif median_days <= 100:
            return "QE"
        else:
            return "YE"

    @staticmethod
    def get_seasonal_period(frequency: str) -> int:
        """Return seasonality period for given frequency."""
        freq_upper = frequency.upper()
        for key, val in FREQ_MAP.items():
            if freq_upper.startswith(key):
                return val
        return 12

    @staticmethod
    def train_test_split(series: pd.Series, ratio: float = 0.8) -> Tuple[pd.Series, pd.Series]:
        """Split time series into train/test sets."""
        n = len(series)
        split = max(int(n * ratio), n - 1)
        return series.iloc[:split], series.iloc[split:]

    @staticmethod
    def parse_csv_bytes(content: bytes, encoding: str = "utf-8") -> pd.DataFrame:
        """Parse CSV from raw bytes (multipart upload)."""
        try:
            return pd.read_csv(io.BytesIO(content), encoding=encoding)
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(content), encoding="latin-1")

    @staticmethod
    def transform_sap_data(
        records: list,
        date_field: str,
        quantity_field: str,
        material_field: Optional[str] = None,
    ) -> pd.DataFrame:
        """Transform SAP/ERP records to standard format."""
        df = pd.DataFrame(records)
        rename = {date_field: "date", quantity_field: "value"}
        if material_field and material_field in df.columns:
            rename[material_field] = "product"
        df = df.rename(columns=rename)
        return df
