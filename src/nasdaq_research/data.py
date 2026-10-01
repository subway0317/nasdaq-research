"""Historical daily market data utilities."""

from collections.abc import Iterable, Mapping
from typing import Any

from pathlib import Path

import pandas as pd

from nasdaq_research.config import (
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
    RAW_DATA_DIR,
    STOCK_SYMBOLS,
)


REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")


def data_path(*parts: str) -> Path:
    """Return a path inside the local data directory."""
    from nasdaq_research.config import DATA_DIR

    return DATA_DIR.joinpath(*parts)


def raw_data_path(symbol: str, raw_dir: Path = RAW_DATA_DIR) -> Path:
    """Return the raw CSV path for a stock symbol."""
    return raw_dir / f"{symbol.upper()}.csv"


def download_stock_history(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """Download and standardize historical daily data for one stock symbol."""
    import yfinance as yf

    raw = yf.download(
        symbol.upper(),
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return standardize_history(raw, symbol=symbol)


def download_stock_pool_history(
    symbols: Iterable[str] = STOCK_SYMBOLS,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    raw_dir: Path = RAW_DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """Download, standardize, and save historical data for a stock pool."""
    results: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        standardized = download_stock_history(symbol, period=period, interval=interval)
        save_stock_history(symbol, standardized, raw_dir=raw_dir)
        results[symbol.upper()] = standardized
    return results


def standardize_history(data: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """Normalize historical OHLCV data to the project CSV schema."""
    if data.empty:
        raise ValueError(f"No historical data returned for {symbol or 'symbol'}")

    frame = _flatten_yfinance_columns(data)
    frame = frame.reset_index()
    frame.columns = [_normalize_column_name(column) for column in frame.columns]

    rename_map = {
        "datetime": "date",
    }
    if "date" not in frame.columns:
        rename_map["index"] = "date"
    frame = frame.rename(columns=rename_map)

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns for {symbol or 'symbol'}: {missing}")

    result = frame.loc[:, REQUIRED_COLUMNS].copy()
    result["date"] = pd.to_datetime(result["date"]).dt.date
    for column in ("open", "high", "low", "close", "volume"):
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result.dropna(subset=list(REQUIRED_COLUMNS))
    result = result.drop_duplicates(subset="date", keep="last")
    result = result.sort_values("date").reset_index(drop=True)

    if result.empty:
        raise ValueError(f"No valid historical rows after standardization for {symbol or 'symbol'}")

    return result


def save_stock_history(
    symbol: str,
    data: pd.DataFrame,
    raw_dir: Path = RAW_DATA_DIR,
) -> Path:
    """Save standardized stock history to a raw CSV file."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_data_path(symbol, raw_dir=raw_dir)
    data.loc[:, REQUIRED_COLUMNS].to_csv(output_path, index=False)
    return output_path


def read_stock_history(symbol: str, raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Read a saved raw stock history CSV and return standardized data."""
    input_path = raw_data_path(symbol, raw_dir=raw_dir)
    data = pd.read_csv(input_path)
    return standardize_history(data, symbol=symbol)


def validate_history(data: pd.DataFrame) -> dict[str, Any]:
    """Return basic data quality checks for standardized history data."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    return {
        "row_count": len(data),
        "columns": list(data.columns),
        "missing_columns": missing_columns,
        "date_monotonic_increasing": data["date"].is_monotonic_increasing
        if "date" in data
        else False,
        "duplicate_dates": int(data["date"].duplicated().sum()) if "date" in data else None,
        "null_counts": data.isna().sum().to_dict(),
        "start_date": data["date"].min() if "date" in data and not data.empty else None,
        "end_date": data["date"].max() if "date" in data and not data.empty else None,
    }


def summarize_saved_histories(
    symbols: Iterable[str] = STOCK_SYMBOLS,
    raw_dir: Path = RAW_DATA_DIR,
) -> dict[str, Mapping[str, Any]]:
    """Summarize saved CSV files for a stock pool."""
    summaries: dict[str, Mapping[str, Any]] = {}
    for symbol in symbols:
        data = read_stock_history(symbol, raw_dir=raw_dir)
        summaries[symbol.upper()] = validate_history(data)
    return summaries


def _flatten_yfinance_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with yfinance MultiIndex columns reduced to field names."""
    frame = data.copy()
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame

    required_names = {"open", "high", "low", "close", "volume"}
    for level in range(frame.columns.nlevels):
        names = {_normalize_column_name(value) for value in frame.columns.get_level_values(level)}
        if required_names.issubset(names):
            frame.columns = frame.columns.get_level_values(level)
            return frame

    frame.columns = ["_".join(str(part) for part in column if part) for column in frame.columns]
    return frame


def _normalize_column_name(column: object) -> str:
    """Normalize a source column name to snake_case."""
    return str(column).strip().lower().replace(" ", "_")
