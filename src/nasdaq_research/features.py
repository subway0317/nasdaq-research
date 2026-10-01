"""Causal daily research features and local processed CSV utilities."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nasdaq_research.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, STOCK_SYMBOLS
from nasdaq_research.data import REQUIRED_COLUMNS, raw_data_path

WARMUP_ROWS = {
    "simple_return": 1, "log_return": 1,
    "intraday_return": 0, "daily_range": 0,
    "sma_5": 4, "sma_20": 19, "sma_60": 59,
    "rolling_volatility_20": 20, "rolling_volatility_60": 60,
}
PROCESSED_COLUMNS = (*REQUIRED_COLUMNS, *WARMUP_ROWS)


def _validate_input(data: pd.DataFrame) -> None:
    """Reject invalid history without dropping, filling, or reordering rows."""
    missing = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("History is empty")
    dates = pd.to_datetime(data["date"], errors="raise")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Missing or duplicate dates")
    if not dates.is_monotonic_increasing:
        raise ValueError("Dates must be in ascending order")
    values = data.loc[:, REQUIRED_COLUMNS[1:]]
    if not all(pd.api.types.is_numeric_dtype(values[col]) for col in values):
        raise ValueError("OHLCV fields must be numeric")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Unexpected missing or non-finite OHLCV data")
    if (values.loc[:, ["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Prices must be positive")
    if (values["volume"] < 0).any():
        raise ValueError("Volume must be nonnegative")


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add current/past-only features, preserving OHLCV and warm-up NaNs."""
    _validate_input(data)
    result = data.loc[:, REQUIRED_COLUMNS].copy()
    close = result["close"]
    result["simple_return"] = close.pct_change(fill_method=None)
    result["log_return"] = np.log(close / close.shift(1))
    result["intraday_return"] = close / result["open"] - 1
    result["daily_range"] = (result["high"] - result["low"]) / close
    for window in (5, 20, 60):
        result[f"sma_{window}"] = close.rolling(window, min_periods=window, center=False).mean()
    for window in (20, 60):
        result[f"rolling_volatility_{window}"] = result["simple_return"].rolling(
            window, min_periods=window, center=False
        ).std(ddof=1)
    quality = validate_features(result)
    if not quality["valid"]:
        raise ValueError(f"Invalid feature data: {quality}")
    return result


def validate_features(data: pd.DataFrame) -> dict[str, Any]:
    """Distinguish expected positional warm-up NaNs from invalid feature values."""
    _validate_input(data)
    checks: dict[str, Any] = {}
    for column, warmup in WARMUP_ROWS.items():
        values = data[column]
        expected = np.arange(len(data)) < warmup
        actual = values.isna().to_numpy()
        checks[column] = {
            "expected_warmup_nan": int(expected.sum()),
            "actual_nan": int(actual.sum()),
            "unexpected_missing": int((actual & ~expected).sum()),
            "missing_warmup_nan": int((expected & ~actual).sum()),
            "infinite_values": int(np.isinf(values.to_numpy(dtype=float)).sum()),
        }
    return {
        "row_count": len(data),
        "start_date": data["date"].iloc[0],
        "end_date": data["date"].iloc[-1],
        "columns": list(data.columns),
        "duplicate_dates": int(pd.to_datetime(data["date"]).duplicated().sum()),
        "date_monotonic_increasing": pd.to_datetime(data["date"]).is_monotonic_increasing,
        "features": checks,
        "valid": all(not (item["unexpected_missing"] or item["missing_warmup_nan"]
                          or item["infinite_values"]) for item in checks.values()),
    }


def processed_data_path(symbol: str, processed_dir: Path = PROCESSED_DATA_DIR) -> Path:
    """Return the local processed CSV path for a symbol."""
    return processed_dir / f"{symbol.upper()}.csv"


def save_processed_history(
    symbol: str, data: pd.DataFrame, processed_dir: Path = PROCESSED_DATA_DIR,
) -> Path:
    """Validate and save processed data with warm-up cells left empty."""
    if not validate_features(data)["valid"]:
        raise ValueError("Unexpected missing or invalid feature values")
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_data_path(symbol, processed_dir)
    data.loc[:, PROCESSED_COLUMNS].to_csv(path, index=False)
    return path


def read_processed_history(
    symbol: str, processed_dir: Path = PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    """Read and validate processed CSV data without filling or dropping rows."""
    data = pd.read_csv(processed_data_path(symbol, processed_dir))
    data["date"] = pd.to_datetime(data["date"]).dt.date
    if not validate_features(data)["valid"]:
        raise ValueError("Unexpected missing or invalid feature values")
    return data


def process_symbol(
    symbol: str, raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    """Build and save one symbol from an existing raw CSV without downloading."""
    raw = pd.read_csv(raw_data_path(symbol, raw_dir))
    raw["date"] = pd.to_datetime(raw["date"]).dt.date
    result = build_features(raw)
    pd.testing.assert_frame_equal(result.loc[:, REQUIRED_COLUMNS], raw.loc[:, REQUIRED_COLUMNS])
    save_processed_history(symbol, result, processed_dir)
    return result


def process_stock_pool(
    symbols: Iterable[str] = STOCK_SYMBOLS, raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """Process the configured stock pool using only local raw CSV files."""
    return {symbol.upper(): process_symbol(symbol, raw_dir, processed_dir) for symbol in symbols}
