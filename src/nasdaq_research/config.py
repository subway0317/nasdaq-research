"""Configuration helpers for the research project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

DEFAULT_PERIOD = "1y"
DEFAULT_INTERVAL = "1d"

STOCK_SYMBOLS = ("NVDA", "AAPL", "MSFT", "AMZN", "GOOGL")

PROCESSED_DATA_DIR = DATA_DIR / "processed"

FUNDAMENTALS_DATA_DIR = DATA_DIR / "fundamentals"
