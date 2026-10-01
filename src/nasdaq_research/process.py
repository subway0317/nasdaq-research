"""Offline entry point for the basic research feature layer."""

from nasdaq_research.features import process_stock_pool, validate_features


def main() -> None:
    """Process local histories and print feature quality summaries."""
    for symbol, data in process_stock_pool().items():
        summary = validate_features(data)
        print(f"{symbol}: {len(data)} rows, {summary['start_date']} to "
              f"{summary['end_date']}, duplicate dates={summary['duplicate_dates']}, "
              f"feature quality valid={summary['valid']}")


if __name__ == "__main__":
    main()
