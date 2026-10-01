import math
import statistics
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nasdaq_research.data import REQUIRED_COLUMNS
from nasdaq_research.features import (
    PROCESSED_COLUMNS, WARMUP_ROWS, build_features, process_stock_pool,
    processed_data_path, read_processed_history, save_processed_history,
    validate_features,
)


def history(rows=75):
    closes = [100.0 + i + (i % 3) * 2 for i in range(rows)]
    return pd.DataFrame({
        "date": pd.bdate_range("2025-01-01", periods=rows).date,
        "open": [c - 1 for c in closes], "high": [c + 2 for c in closes],
        "low": [c - 3 for c in closes], "close": closes,
        "volume": [1000 + i for i in range(rows)],
    })


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.raw = history()
        self.result = build_features(self.raw)

    def test_simple_return(self):
        for i in range(1, len(self.raw)):
            self.assertAlmostEqual(self.result.simple_return.iloc[i],
                                   self.raw.close.iloc[i] / self.raw.close.iloc[i - 1] - 1)

    def test_log_return(self):
        for i in range(1, len(self.raw)):
            self.assertAlmostEqual(self.result.log_return.iloc[i],
                                   math.log(self.raw.close.iloc[i] / self.raw.close.iloc[i - 1]))

    def test_intraday_return(self):
        for i in range(len(self.raw)):
            self.assertAlmostEqual(self.result.intraday_return.iloc[i],
                                   self.raw.close.iloc[i] / self.raw.open.iloc[i] - 1)

    def test_daily_range(self):
        for i in range(len(self.raw)):
            self.assertAlmostEqual(self.result.daily_range.iloc[i], 5 / self.raw.close.iloc[i])

    def test_sma_windows(self):
        for window in (5, 20, 60):
            for i in range(window - 1, len(self.raw)):
                expected = statistics.mean(self.raw.close.iloc[i - window + 1:i + 1])
                self.assertAlmostEqual(self.result[f"sma_{window}"].iloc[i], expected)

    def test_daily_sample_volatility(self):
        returns = [self.raw.close.iloc[i] / self.raw.close.iloc[i - 1] - 1
                   for i in range(1, len(self.raw))]
        for window in (20, 60):
            for i in range(window, len(self.raw)):
                expected = statistics.stdev(returns[i - window:i])
                self.assertAlmostEqual(self.result[f"rolling_volatility_{window}"].iloc[i], expected)

    def test_warmup_nan_including_short_history(self):
        for rows in (1, 4, 20, 60, 75):
            result = build_features(history(rows))
            for column, count in WARMUP_ROWS.items():
                self.assertEqual(result[column].isna().tolist(),
                                 [i < count for i in range(rows)])
            self.assertTrue(validate_features(result)["valid"])

    def test_preserves_dates_ohlcv_and_input(self):
        original = self.raw.copy(deep=True)
        result = build_features(self.raw)
        pd.testing.assert_frame_equal(result.loc[:, REQUIRED_COLUMNS], original)
        pd.testing.assert_frame_equal(self.raw, original)
        self.assertEqual(list(result.columns), list(PROCESSED_COLUMNS))
        self.assertFalse(result.date.duplicated().any())

    def test_rejects_duplicate_and_unordered_dates(self):
        duplicate = self.raw.copy()
        duplicate.loc[1, "date"] = duplicate.loc[0, "date"]
        for data in (duplicate, self.raw.iloc[::-1]):
            with self.assertRaises(ValueError):
                build_features(data)

    def test_rejects_invalid_raw_without_silent_cleaning(self):
        for column, value in (("close", float("nan")), ("open", 0),
                              ("high", float("inf")), ("volume", -1)):
            raw = self.raw.copy()
            raw.loc[2, column] = value
            with self.assertRaises(ValueError):
                build_features(raw)
        with self.assertRaises(ValueError):
            build_features(self.raw.drop(columns="close"))
        with self.assertRaises(ValueError):
            build_features(self.raw.iloc[:0])

    def test_unexpected_missing_and_infinity_are_reported(self):
        result = self.result.copy()
        result.loc[70, "sma_20"] = float("nan")
        result.loc[71, "log_return"] = float("inf")
        quality = validate_features(result)
        self.assertFalse(quality["valid"])
        self.assertEqual(quality["features"]["sma_20"]["unexpected_missing"], 1)
        self.assertEqual(quality["features"]["log_return"]["infinite_values"], 1)
        result.loc[0, "simple_return"] = 0.0
        self.assertEqual(validate_features(result)["features"]["simple_return"]["missing_warmup_nan"], 1)

    def test_no_lookahead_prefix_and_future_mutation(self):
        for stop in (1, 5, 20, 21, 60, 61, 70):
            pd.testing.assert_frame_equal(build_features(self.raw.iloc[:stop]),
                                          self.result.iloc[:stop])
        changed = self.raw.copy()
        changed.loc[65:, ["open", "high", "low", "close"]] *= 3
        changed.loc[65:, "volume"] *= 2
        pd.testing.assert_frame_equal(build_features(changed).iloc[:65], self.result.iloc[:65])

    def test_processed_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = save_processed_history("test", self.result, folder)
            loaded = read_processed_history("TEST", folder)
            self.assertEqual(path, processed_data_path("test", folder))
            pd.testing.assert_frame_equal(loaded, self.result)

    def test_offline_pool_and_raw_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir, processed_dir = root / "raw", root / "processed"
            raw_dir.mkdir()
            for symbol in ("ONE", "TWO"):
                self.raw.to_csv(raw_dir / f"{symbol}.csv", index=False)
            before = {path.name: path.read_bytes() for path in raw_dir.iterdir()}
            results = process_stock_pool(("ONE", "TWO"), raw_dir, processed_dir)
            self.assertEqual(set(results), {"ONE", "TWO"})
            self.assertEqual(before, {path.name: path.read_bytes() for path in raw_dir.iterdir()})
            for symbol, result in results.items():
                self.assertEqual(len(result), len(self.raw))
                pd.testing.assert_frame_equal(read_processed_history(symbol, processed_dir), result)


if __name__ == "__main__":
    unittest.main()
