"""Offline Stage 6 formula, period matching, vintage and preservation tests."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from nasdaq_research.alignment import build_research_dataset
from nasdaq_research.features import WARMUP_ROWS, build_features
from nasdaq_research.fundamental_mapping import FINANCIAL_FIELDS
from nasdaq_research.research_features import (
    ANNUAL_GROWTH_FEATURES, ANNUAL_RATIO_FEATURES, BALANCE_RATIOS,
    FEATURE_COLUMNS, GROWTH_FIELDS, PAYLOAD_COLUMNS, QUARTERLY_GROWTH_FEATURES,
    QUARTERLY_RATIO_FEATURES, build_feature_matrix, build_fundamental_features,
    run_pipeline, validate_feature_matrix,
)


def observation(fy=2027, fp="Q2", basis="quarterly", revenue=200.0, **changes):
    year = fy - 1
    periods = {
        "Q1": (f"{year}-02-01", f"{year}-04-30", f"{year}-05-20"),
        "Q2": (f"{year}-05-01", f"{year}-07-31", f"{year}-08-26"),
        "Q3": (f"{year}-08-01", f"{year}-10-31", f"{year}-11-20"),
        "Q4": (f"{year}-11-01", f"{fy}-01-31", f"{fy}-02-25"),
        "FY": (f"{year}-02-01", f"{fy}-01-31", f"{fy}-02-25"),
    }
    start, end, filed = periods[fp]
    row = {"ticker": "NVDA", "fiscal_year": fy, "fiscal_period": fp,
           "period_start": start, "period_end": end, "filing_date": filed,
           "form": "10-K" if basis == "annual" or fp == "Q4" else "10-Q",
           "accession": f"{fy}-{fp}-{basis}-{filed}", "data_period_type": basis,
           **{c: np.nan for c in FINANCIAL_FIELDS},
           "revenue": revenue, "gross_profit": revenue * .6,
           "operating_income": revenue * .3, "net_income": revenue * .2,
           "research_and_development": revenue * .1,
           "selling_general_administrative": revenue * .05,
           "operating_cash_flow": revenue * .25, "capital_expenditures": abs(revenue) * .05,
           "free_cash_flow": revenue * .25 - abs(revenue) * .05,
           "current_assets": 180., "current_liabilities": 90.,
           "cash_and_cash_equivalents": 30., "total_assets": 300.,
           "total_liabilities": 150., "long_term_debt": 60., "revenue_concept": "Revenues"}
    row.update(changes)
    if "operating_cash_flow" in changes or "capital_expenditures" in changes:
        ocf, capex = row["operating_cash_flow"], row["capital_expenditures"]
        row["free_cash_flow"] = ocf - capex if pd.notna(ocf) and pd.notna(capex) else np.nan
    return row


def market(dates=None):
    dates = dates if dates is not None else pd.bdate_range("2026-08-24", "2026-09-04")
    close = np.arange(len(dates), dtype=float) + 100
    return build_features(pd.DataFrame({
        "date": pd.to_datetime(dates).date, "open": close - 1, "high": close + 2,
        "low": close - 2, "close": close, "volume": np.arange(len(dates)) + 1000,
    }))


def build(rows, prices=None):
    f = pd.DataFrame(rows)
    r = build_research_dataset(market() if prices is None else prices, f)
    s, d = build_feature_matrix(r, f)
    return f, r, s, d


def snapshot(rows, accession=None):
    result = build_fundamental_features(pd.DataFrame(rows))
    return result.iloc[-1] if accession is None else result[result.accession.eq(accession)].iloc[0]


class RatioTests(unittest.TestCase):
    def test_quarterly_ratios(self):
        row = snapshot([observation()])
        for feature, value in zip(QUARTERLY_RATIO_FEATURES, (.6, .3, .2, .1, .05, .25, .2)):
            self.assertAlmostEqual(row[feature], value, msg=feature)
        self.assertTrue(row[list(ANNUAL_RATIO_FEATURES)].isna().all())

    def test_annual_ratios(self):
        row = snapshot([observation(fp="FY", basis="annual")])
        for feature, value in zip(ANNUAL_RATIO_FEATURES, (.6, .3, .2, .1, .05, .25, .2)):
            self.assertAlmostEqual(row[feature], value, msg=feature)
        self.assertTrue(row[list(QUARTERLY_RATIO_FEATURES)].isna().all())

    def test_zero_missing_and_negative_revenue(self):
        for value in (0., np.nan, -1.):
            for fp, basis, columns in (("Q2", "quarterly", QUARTERLY_RATIO_FEATURES),
                                       ("FY", "annual", ANNUAL_RATIO_FEATURES)):
                row = snapshot([observation(fp=fp, basis=basis, revenue=value)])
                self.assertTrue(row[list(columns)].isna().all())

    def test_missing_numerator(self):
        row = snapshot([observation(gross_profit=np.nan, operating_cash_flow=np.nan)])
        self.assertTrue(pd.isna(row.q_gross_margin))
        self.assertTrue(pd.isna(row.q_ocf_margin))
        self.assertTrue(pd.isna(row.q_fcf_margin))
        self.assertAlmostEqual(row.q_net_margin, .2)

    def test_balance_sheet_ratios(self):
        for basis, fp in (("quarterly", "Q2"), ("annual", "FY")):
            row = snapshot([observation(basis=basis, fp=fp)])
            for feature, value in zip(BALANCE_RATIOS, (2., .1, .5, .2)):
                self.assertAlmostEqual(row[feature], value, msg=feature)

    def test_balance_zero_and_missing_denominator(self):
        for value in (0., np.nan):
            row = snapshot([observation(current_liabilities=value, total_assets=value)])
            self.assertTrue(row[list(BALANCE_RATIOS)].isna().all())

    def test_real_zero_is_retained(self):
        row = snapshot([observation(net_income=0., long_term_debt=0.)])
        self.assertEqual(row.q_net_margin, 0.)
        self.assertEqual(row.debt_to_assets, 0.)

    def test_overflow_stays_nan(self):
        row = snapshot([observation(revenue=1e-300, gross_profit=1e300)])
        self.assertTrue(pd.isna(row.q_gross_margin))
        self.assertFalse(np.isinf(row[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).any())


class GrowthTests(unittest.TestCase):
    def test_q2_yoy_uses_same_quarter_previous_fy(self):
        rows = [observation(fy=2026, revenue=100), observation(fp="Q1", revenue=50), observation()]
        row = snapshot(rows)
        for feature in QUARTERLY_GROWTH_FEATURES:
            self.assertAlmostEqual(row[feature], 1., msg=feature)
        self.assertEqual(row.yoy_reference_fiscal_year, 2026)
        self.assertEqual(row.yoy_reference_fiscal_period, "Q2")

    def test_all_four_fiscal_quarters(self):
        for fp in ("Q1", "Q2", "Q3", "Q4"):
            row = snapshot([observation(fy=2026, fp=fp, revenue=100), observation(fp=fp)])
            self.assertAlmostEqual(row.q_revenue_growth_yoy, 1.)
            self.assertEqual(row.yoy_reference_fiscal_period, fp)

    def test_q1_is_not_a_q2_comparator(self):
        for fy in (2026, 2027):
            row = snapshot([observation(fy=fy, fp="Q1", revenue=100), observation()])
            self.assertTrue(row[list(QUARTERLY_GROWTH_FEATURES)].isna().all())
            self.assertEqual(row.yoy_comparison_status, "missing_previous_period")

    def test_annual_previous_year(self):
        row = snapshot([observation(fy=2026, fp="FY", basis="annual", revenue=100),
                        observation(fp="FY", basis="annual")])
        for feature in ANNUAL_GROWTH_FEATURES:
            self.assertAlmostEqual(row[feature], 1., msg=feature)
        self.assertEqual(row.yoy_reference_fiscal_period, "FY")

    def test_missing_previous_year_and_skipped_year(self):
        for basis, fp, columns in (("quarterly", "Q2", QUARTERLY_GROWTH_FEATURES),
                                   ("annual", "FY", ANNUAL_GROWTH_FEATURES)):
            row = snapshot([observation(fy=2025, fp=fp, basis=basis), observation(fp=fp, basis=basis)])
            self.assertTrue(row[list(columns)].isna().all())

    def test_growth_zero_or_negative_denominator(self):
        for denominator in (0., -10.):
            for basis, fp, columns in (("quarterly", "Q2", QUARTERLY_GROWTH_FEATURES),
                                       ("annual", "FY", ANNUAL_GROWTH_FEATURES)):
                previous = observation(fy=2026, fp=fp, basis=basis,
                                       **{field: denominator for field in GROWTH_FIELDS.values()})
                row = snapshot([previous, observation(fp=fp, basis=basis)])
                self.assertTrue(row[list(columns)].isna().all())

    def test_missing_growth_inputs_are_not_filled(self):
        row = snapshot([observation(fy=2026, net_income=np.nan), observation(gross_profit=np.nan)])
        self.assertTrue(pd.isna(row.q_net_income_growth_yoy))
        self.assertTrue(pd.isna(row.q_gross_profit_growth_yoy))
        self.assertAlmostEqual(row.q_revenue_growth_yoy, 0.)

    def test_no_quarterly_annual_cross_comparison(self):
        row = snapshot([observation(fy=2026, fp="FY", basis="annual"), observation()])
        self.assertTrue(row[list(QUARTERLY_GROWTH_FEATURES)].isna().all())
        annual = snapshot([observation(fy=2026, fp="Q4"), observation(fp="FY", basis="annual")])
        self.assertTrue(annual[list(ANNUAL_GROWTH_FEATURES)].isna().all())

    def test_latest_known_amendment_and_frozen_reference(self):
        previous = observation(fy=2026, revenue=100, accession="prior-original")
        known = observation(fy=2026, revenue=120, accession="prior-known", form="10-Q/A", filing_date="2026-08-20")
        future = observation(fy=2026, revenue=999, accession="prior-future", form="10-Q/A", filing_date="2026-09-01")
        current = observation(revenue=180, accession="current")
        amended = observation(revenue=240, accession="current-amended", form="10-Q/A", filing_date="2026-09-02")
        rows = [previous, known, current, future, amended]
        row = snapshot(rows, "current")
        self.assertAlmostEqual(row.q_revenue_growth_yoy, .5)
        self.assertEqual(row.yoy_reference_accession, "prior-known")
        later = snapshot(rows, "current-amended")
        self.assertAlmostEqual(later.q_revenue_growth_yoy, 240 / 999 - 1)
        self.assertEqual(later.yoy_reference_accession, "prior-future")

    def test_missing_latest_version_does_not_fallback(self):
        previous = observation(fy=2026, revenue=100)
        amended = observation(fy=2026, revenue=np.nan, accession="missing", form="10-Q/A", filing_date="2026-08-20")
        row = snapshot([previous, amended, observation()])
        self.assertTrue(pd.isna(row.q_revenue_growth_yoy))
        self.assertEqual(row.yoy_reference_accession, "missing")

    def test_matching_labels_with_wrong_dates_are_rejected(self):
        previous = observation(fy=2026, revenue=100, period_start="2025-08-01", period_end="2025-10-31", filing_date="2025-11-20")
        row = snapshot([previous, observation()])
        self.assertEqual(row.yoy_comparison_status, "noncomparable_period_dates")
        self.assertTrue(pd.isna(row.q_revenue_growth_yoy))

    def test_ambiguous_fiscal_key_is_not_arbitrarily_matched(self):
        prior = observation(fy=2026)
        bad = observation(fy=2026, period_start="2025-08-01", period_end="2025-10-31", filing_date="2025-11-20", accession="bad-label")
        result = build_fundamental_features(pd.DataFrame([prior, bad, observation()]))
        self.assertEqual(result.iloc[1].yoy_comparison_status, "ambiguous_current_period")
        self.assertEqual(result.iloc[2].yoy_comparison_status, "ambiguous_previous_period")
        self.assertTrue(result.iloc[2][list(QUARTERLY_GROWTH_FEATURES)].isna().all())

    def test_future_duplicate_label_does_not_change_past_snapshot(self):
        previous = observation(fy=2026, revenue=100)
        current = observation(accession="current")
        bad = observation(fy=2026, period_start="2025-08-01", period_end="2025-10-31", filing_date="2026-09-01", accession="future-label")
        pd.testing.assert_series_equal(snapshot([previous, current], "current"),
                                       snapshot([previous, current, bad], "current"), check_names=False)


class PointInTimeTests(unittest.TestCase):
    def test_filing_day_next_day_and_early_missing(self):
        f, r, s, d = build([observation()])
        self.assertTrue(d.loc[:2, list(FEATURE_COLUMNS)].isna().all().all())
        self.assertAlmostEqual(d.q_gross_margin.iloc[3], .6)
        self.assertEqual(d.fundamental_effective_date.iloc[3], pd.Timestamp("2026-08-27"))
        self.assertTrue(validate_feature_matrix(d, s, r, f)["valid"])

    def test_holiday_calendar_matches_stage5(self):
        prices = market(["2026-08-26", "2026-08-28", "2026-08-31"])
        _, r, s, d = build([observation()], prices)
        self.assertTrue(pd.isna(d.q_gross_margin.iloc[0]))
        self.assertAlmostEqual(d.q_gross_margin.iloc[1], .6)
        self.assertEqual(s.effective_date.iloc[0], pd.Timestamp("2026-08-28"))
        pd.testing.assert_frame_equal(d[r.columns], r)

    def test_future_filing_and_future_market_mutations(self):
        prices = market()
        rows = [observation(), observation(accession="future", filing_date="2026-09-02", gross_profit=80)]
        _, r, s, d = build(rows, prices)
        rows[1]["gross_profit"] = 1000
        changed = prices.copy()
        changed.loc[8:, "close"] *= 2
        # Change raw prices through the Stage 3 interface; earlier market features stay causal.
        raw_columns = ["date", "open", "high", "low", "close", "volume"]
        changed = build_features(changed[raw_columns])
        _, _, _, later = build(rows, changed)
        pd.testing.assert_frame_equal(d.iloc[:8], later.iloc[:8])

    def test_prefix_invariance(self):
        f, r, _, d = build([observation(fy=2026), observation(),
                            observation(accession="future", filing_date="2026-09-02")])
        for stop in (1, 3, 4, 8):
            _, prefix = build_feature_matrix(r.iloc[:stop], f)
            pd.testing.assert_frame_equal(d.iloc[:stop], prefix)

    def test_ytd_and_missing_quarterly_cash_flow(self):
        current = observation(operating_cash_flow=np.nan, capital_expenditures=np.nan)
        ytd = observation(basis="ytd", period_start="2026-02-01", revenue=1000)
        f, r, s, d = build([current, ytd])
        self.assertEqual(len(s), 1)
        self.assertTrue(d.q_ocf_margin.isna().all())
        self.assertTrue(d.q_fcf_margin.isna().all())
        self.assertAlmostEqual(d.q_gross_margin.iloc[3], .6)
        self.assertTrue(validate_feature_matrix(d, s, r, f)["valid"])

    def test_all_ytd_or_all_future(self):
        for row in (observation(basis="ytd", period_start="2026-02-01"),
                    observation(filing_date="2026-10-01")):
            f, r, s, d = build([row])
            self.assertTrue(d[list(FEATURE_COLUMNS)].isna().all().all())
            self.assertTrue(validate_feature_matrix(d, s, r, f)["valid"])

    def test_deterministic_order_and_no_input_mutation(self):
        f, r, s, d = build([observation(fy=2026), observation()])
        original_f, original_r = f.copy(deep=True), r.copy(deep=True)
        for seed in range(3):
            shuffled_s, shuffled_d = build_feature_matrix(r, f.sample(frac=1, random_state=seed))
            pd.testing.assert_frame_equal(s, shuffled_s)
            pd.testing.assert_frame_equal(d, shuffled_d)
        pd.testing.assert_frame_equal(f, original_f)
        pd.testing.assert_frame_equal(r, original_r)

    def test_same_filing_q4_priority_uses_whole_quarter_snapshot(self):
        current = observation(fp="Q4", accession="same-filing")
        annual = observation(fp="FY", basis="annual", accession="same-filing", revenue=1000)
        _, _, s, d = build([current, annual], market(pd.bdate_range("2027-02-24", "2027-03-01")))
        self.assertEqual(int(s.selected_for_daily.sum()), 2)
        self.assertEqual(int(s.selected_for_stage5.sum()), 1)
        self.assertEqual(d.fundamental_data_period_type.iloc[2], "quarterly")
        self.assertAlmostEqual(d.fy_gross_margin.iloc[2], .6)
        self.assertEqual(d.fy_accession.iloc[2], "same-filing")
        self.assertEqual(d.revenue.iloc[2], 200)


class OutputTests(unittest.TestCase):
    def test_expected_columns_row_count_market_preservation_and_no_infinity(self):
        f, r, s, d = build([observation(fy=2026), observation()])
        self.assertEqual(len(FEATURE_COLUMNS), 30)
        self.assertEqual(len(d), len(r))
        self.assertTrue(set(FEATURE_COLUMNS).issubset(d))
        pd.testing.assert_frame_equal(d[r.columns], r)
        self.assertFalse(np.isinf(d.select_dtypes("number").to_numpy(dtype=float)).any())
        self.assertTrue(validate_feature_matrix(d, s, r, f)["valid"])

    def test_validation_rejects_wrong_values_periods_lineage_and_leakage(self):
        f, r, s, d = build([observation(fy=2026), observation()])
        for column, value in (("q_gross_margin", .123), ("q_revenue_growth_yoy", 99.),
                              ("fy_gross_margin", .1), ("q_ocf_margin", np.inf),
                              ("fundamental_filing_date", "2026-09-04"),
                              ("yoy_reference_accession", "wrong"), ("close", 999.), ("ticker", "AAPL")):
            bad = d.copy(); bad.loc[3, column] = value
            self.assertFalse(validate_feature_matrix(bad, s, r, f)["valid"], column)
        self.assertFalse(validate_feature_matrix(d.iloc[1:], s, r, f)["valid"])
        bad_s = s.copy(); bad_s.loc[1, "yoy_reference_fiscal_period"] = "Q1"
        self.assertFalse(validate_feature_matrix(d, bad_s, r, f)["valid"])
        bad_s = s.copy(); bad_s.loc[1, "yoy_reference_filing_date"] = "2026-09-04"
        self.assertFalse(validate_feature_matrix(d, bad_s, r, f)["valid"])

    def test_validation_rejects_string_features_and_duplicate_dates(self):
        f, r, s, d = build([observation()])
        bad = d.copy().astype({"q_gross_margin": object}); bad.loc[3, "q_gross_margin"] = "0.6"
        self.assertFalse(validate_feature_matrix(bad, s, r, f)["valid"])
        r.loc[1, "date"] = r.loc[0, "date"]
        with self.assertRaises(ValueError):
            build_feature_matrix(r, f)

    def test_invalid_stage5_metadata_is_rejected(self):
        f, r, _, _ = build([observation()])
        r.loc[3, "fundamental_accession"] = "wrong"
        with self.assertRaisesRegex(ValueError, "Stage 5"):
            build_feature_matrix(r, f)

    def test_pipeline_and_cli_offline_roundtrip_preserve_inputs(self):
        f, r, _, _ = build([observation(fy=2026), observation()])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fpath = root / "fundamentals.csv"; rpath = root / "research.csv"
            f.to_csv(fpath, index=False); r.to_csv(rpath, index=False)
            before = (fpath.read_bytes(), rpath.read_bytes())
            with patch("nasdaq_research.fundamentals.download_company_facts", side_effect=AssertionError("network forbidden")), \
                 patch("nasdaq_research.data.download_stock_history", side_effect=AssertionError("network forbidden")):
                summary = run_pipeline(fpath, rpath, root)
            self.assertTrue(summary["validation"]["valid"])
            self.assertEqual(summary["new_feature_count"], 30)
            self.assertEqual(summary["engineered_feature_count"], 30 + len(WARMUP_ROWS))
            loaded = pd.read_csv(summary["final_feature_matrix_path"])
            self.assertEqual(len(loaded), len(r))
            self.assertTrue(set(PAYLOAD_COLUMNS).issubset(loaded))
            self.assertEqual(json.loads((root / "NVDA_feature_validation.json").read_text())["validation"]["valid"], True)
            command = [sys.executable, "-m", "nasdaq_research.research_features",
                       "--fundamentals-path", str(fpath), "--research-path", str(rpath), "--output-dir", str(root)]
            run = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertIn('"final_feature_rows": 10', run.stdout)
            self.assertEqual(before, (fpath.read_bytes(), rpath.read_bytes()))
            with self.assertRaisesRegex(ValueError, "overwrite"):
                run_pipeline(fpath, root / "NVDA_features.csv", root)


if __name__ == "__main__":
    unittest.main()
