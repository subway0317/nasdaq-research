import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nasdaq_research.config import STOCK_SYMBOLS
from nasdaq_research.data import (
    REQUIRED_COLUMNS,
    data_path,
    raw_data_path,
    read_stock_history,
    save_stock_history,
    standardize_history,
)


class DataTests(unittest.TestCase):
    def test_stock_pool_configuration(self):
        self.assertEqual(STOCK_SYMBOLS, ("NVDA", "AAPL", "MSFT", "AMZN", "GOOGL"))

    def test_data_paths(self):
        self.assertTrue(str(data_path("raw")).endswith("data/raw"))
        self.assertEqual(raw_data_path("nvda").name, "NVDA.csv")

    def test_standardize_history(self):
        raw = pd.DataFrame(
            {
                "Date": ["2026-01-03", "2026-01-01", "2026-01-01"],
                "Open": ["3.0", "1.0", "1.1"],
                "High": [3.5, 1.5, 1.6],
                "Low": [2.5, 0.5, 0.6],
                "Close": [3.25, 1.25, 1.35],
                "Volume": [300, 100, 110],
                "Adj Close": [3.2, 1.2, 1.3],
            }
        )

        result = standardize_history(raw, symbol="TEST")

        self.assertEqual(list(result.columns), list(REQUIRED_COLUMNS))
        self.assertEqual(result["date"].tolist(), [pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-01-03").date()])
        self.assertFalse(result["date"].duplicated().any())
        self.assertTrue(pd.api.types.is_numeric_dtype(result["open"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(result["volume"]))

    def test_save_and_read_stock_history(self):
        data = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-01-02").date()],
                "open": [1.0, 2.0],
                "high": [1.5, 2.5],
                "low": [0.5, 1.5],
                "close": [1.25, 2.25],
                "volume": [1000, 2000],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            output_path = save_stock_history("TEST", data, raw_dir=raw_dir)
            loaded = read_stock_history("TEST", raw_dir=raw_dir)

        self.assertEqual(output_path.name, "TEST.csv")
        pd.testing.assert_frame_equal(loaded, data)


if __name__ == "__main__":
    unittest.main()
