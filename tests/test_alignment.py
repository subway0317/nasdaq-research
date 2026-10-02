"""Offline Stage 5 leakage and source integrity regression tests."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nasdaq_research.alignment import build_research_dataset, run_pipeline, validate_research_dataset
from nasdaq_research.features import build_features, save_processed_history
from nasdaq_research.fundamental_mapping import FINANCIAL_FIELDS
from test_features import history


def observation(filed="2025-05-28", accession="original", value=100, **changes):
    row = {"ticker": "NVDA", "fiscal_year": 2026, "fiscal_period": "Q1",
           "period_start": "2025-01-27", "period_end": "2025-04-27",
           "filing_date": filed, "form": "10-Q", "accession": accession,
           "data_period_type": "quarterly", **{c: float("nan") for c in FINANCIAL_FIELDS},
           "revenue": value, "revenue_concept": "Revenues"}
    row.update(changes)
    return row


def market():
    return pd.DataFrame({"date": pd.to_datetime([
        "2025-05-27", "2025-05-28", "2025-05-29", "2025-05-30",
        "2025-06-02", "2025-06-03", "2025-06-04",
    ]).date, "close": range(7), "custom_feature": [float("nan"), 1, 2, 3, 4, 5, 6]})


class AlignmentTests(unittest.TestCase):
    def test_before_same_day_next_day_and_forward_hold(self):
        m = market(); f = pd.DataFrame([observation()])
        result = build_research_dataset(m, f)
        self.assertTrue(result.revenue.iloc[:2].isna().all())
        self.assertEqual(result.revenue.iloc[2:].tolist(), [100] * 5)
        self.assertEqual(result.days_since_filing.iloc[2:].tolist(), [1, 2, 5, 6, 7])
        self.assertEqual(result.days_since_effective_date.iloc[2], 0)
        pd.testing.assert_frame_equal(result[list(m)], m)
        self.assertTrue(validate_research_dataset(result, m, f)["valid"])

    def test_weekend_and_missing_holiday_session(self):
        m = market().drop(index=4).reset_index(drop=True)  # June 2 absent from calendar
        for filing in ("2025-05-30", "2025-05-31", "2025-06-01", "2025-06-02"):
            result = build_research_dataset(m, pd.DataFrame([observation(filing)]))
            self.assertTrue(result.revenue.iloc[:4].isna().all())
            self.assertEqual(result.fundamental_effective_date.iloc[4], pd.Timestamp("2025-06-03"))

    def test_amendment_changes_only_after_its_filing(self):
        f = pd.DataFrame([observation(), observation("2025-06-02", "amend", 200, form="10-Q/A")])
        r = build_research_dataset(market(), f)
        self.assertEqual(r.revenue.iloc[2:].tolist(), [100, 100, 100, 200, 200])
        self.assertEqual(r.fundamental_accession.iloc[4], "original")
        self.assertEqual(r.fundamental_accession.iloc[5], "amend")

    def test_ytd_excluded_and_no_field_level_fill(self):
        f = pd.DataFrame([observation(), observation("2025-06-02", "q2", float("nan")),
                          observation("2025-06-02", "ytd", 999, fiscal_period="Q2",
                                      period_start="2024-11-01", data_period_type="ytd")])
        r = build_research_dataset(market(), f)
        self.assertTrue(r.revenue.iloc[5:].isna().all())
        self.assertEqual(r.fundamental_accession.iloc[5], "q2")

    def test_same_day_priority_is_deterministic(self):
        annual = observation(value=900, accession="annual", period_start="2024-04-28",
                             fiscal_period="FY", form="10-K", data_period_type="annual")
        rows = [annual, observation(accession="a"), observation(accession="z", value=300, form="10-Q/A")]
        expected = build_research_dataset(market(), pd.DataFrame(rows))
        self.assertEqual(expected.revenue.iloc[2], 300)
        for seed in range(4):
            pd.testing.assert_frame_equal(expected, build_research_dataset(market(), pd.DataFrame(rows).sample(frac=1, random_state=seed)))

    def test_future_mutation_and_prefix_invariance(self):
        f = pd.DataFrame([observation(), observation("2025-06-03", "future", 500)])
        baseline = build_research_dataset(market(), f)
        f.loc[1, "revenue"] = 900
        pd.testing.assert_frame_equal(baseline.iloc[:6], build_research_dataset(market(), f).iloc[:6])
        pd.testing.assert_frame_equal(baseline.iloc[:4], build_research_dataset(market().iloc[:4], f))

    def test_all_future_or_ytd_and_left_boundary(self):
        for row in (observation("2025-06-05"), observation(fiscal_period="Q2", period_start="2024-11-01", data_period_type="ytd")):
            r = build_research_dataset(market(), pd.DataFrame([row]))
            self.assertTrue(r.revenue.isna().all())
            self.assertTrue(r.days_since_filing.isna().all())
        r = build_research_dataset(market(), pd.DataFrame([observation("2025-05-01")]))
        self.assertEqual(r.fundamental_effective_date.iloc[0], pd.Timestamp("2025-05-27"))

    def test_reject_bad_inputs(self):
        m = market(); f = pd.DataFrame([observation()])
        for bad in (m.iloc[::-1], pd.concat([m, m.iloc[:1]]), m.assign(ticker="AAPL"), m.assign(revenue=0)):
            with self.assertRaises(ValueError):
                build_research_dataset(bad, f)
        with self.assertRaises(ValueError):
            build_research_dataset(m, pd.concat([f, f]))

    def test_validation_detects_leakage_row_loss_and_mixed_observation(self):
        m = market(); f = pd.DataFrame([observation()]); r = build_research_dataset(m, f)
        for column, value in (("revenue", 999), ("fundamental_accession", "wrong"),
                              ("fundamental_filing_date", "2025-06-05"),
                              ("fundamental_effective_date", pd.Timestamp("2025-06-05")),
                              ("days_since_filing", -1), ("close", 99), ("ticker", "AAPL")):
            bad = r.copy(); bad.loc[2, column] = value
            self.assertFalse(validate_research_dataset(bad, m, f)["valid"])
        self.assertFalse(validate_research_dataset(r.iloc[1:], m, f)["valid"])

    def test_local_pipeline_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); prices = root / "market"; fpath = root / "fundamentals.csv"
            save_processed_history("NVDA", build_features(history()), prices)
            pd.DataFrame([observation("2025-05-28")]).to_csv(fpath, index=False)
            before = (fpath.read_bytes(), (prices / "NVDA.csv").read_bytes())
            report = run_pipeline(prices, fpath, root / "research")
            self.assertTrue(report["valid"])
            self.assertEqual(report["row_count"], 75)
            self.assertEqual(before, (fpath.read_bytes(), (prices / "NVDA.csv").read_bytes()))
            self.assertTrue((root / "research" / "NVDA_research.csv").exists())
