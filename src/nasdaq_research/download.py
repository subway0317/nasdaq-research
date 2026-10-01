"""Simple module entry point for downloading the configured stock pool."""

from nasdaq_research.config import STOCK_SYMBOLS
from nasdaq_research.data import download_stock_pool_history, validate_history


def main() -> None:
    """Download the configured stock pool and print a compact summary."""
    results = download_stock_pool_history()
    for symbol in STOCK_SYMBOLS:
        summary = validate_history(results[symbol])
        print(
            f"{symbol}: {summary['row_count']} rows, "
            f"{summary['start_date']} to {summary['end_date']}"
        )


if __name__ == "__main__":
    main()
