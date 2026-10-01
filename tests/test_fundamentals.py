"""Offline synthetic Company Facts tests; no live SEC calls."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from nasdaq_research.fundamentals import (
    CIK, ENDPOINT, download_company_facts, normalize_company_facts,
    run_pipeline, validate_fundamentals,
)


def fact(value=100, start="2024-01-29", end="2025-01-26", **changes):
    result = {"val": value, "start": start, "end": end, "filed": "2025-02-20",
              "fy": 2025, "fp": "FY", "form": "10-K", "accn": "annual-2025"}
    result.update(changes)
    if start is None:
        result.pop("start")
    return result


def payload(concepts=None):
    concepts = concepts or {
        "RevenueFromContractWithCustomerExcludingAssessedTax": [fact(100)],
        "NetCashProvidedByUsedInOperatingActivities": [fact(30)],
        "PaymentsToAcquirePropertyPlantAndEquipment": [fact(10)],
        "Assets": [fact(80, start=None)],
    }
    return {"cik": CIK, "entityName": "NVIDIA CORP", "facts": {"us-gaap": {
        concept: {"units": {"USD": records}} for concept, records in concepts.items()
    }}}


class FundamentalTests(unittest.TestCase):
    def test_mapping_annual_and_fcf(self):
        data, report = normalize_company_facts(payload())
        self.assertEqual(len(data), 1)
        self.assertEqual(data.iloc[0].data_period_type, "annual")
        self.assertEqual(data.iloc[0].fiscal_year, 2025)
        self.assertEqual(data.iloc[0].revenue, 100)
        self.assertEqual(data.iloc[0].total_assets, 80)
        self.assertEqual(data.iloc[0].free_cash_flow, 20)
        self.assertEqual(report["issues"], [])
        self.assertTrue(validate_fundamentals(data)["valid"])

    def test_candidates_priority_and_fallback_per_period(self):
        data, _ = normalize_company_facts(payload({
            "RevenueFromContractWithCustomerExcludingAssessedTax": [fact(100)],
            "SalesRevenueNet": [fact(99)], "Revenues": [fact(98)],
        }))
        self.assertEqual(data.iloc[0].revenue, 100)
        self.assertEqual(data.iloc[0].revenue_concept, "RevenueFromContractWithCustomerExcludingAssessedTax")
        data, _ = normalize_company_facts(payload({"SalesRevenueNet": [fact(99)]}))
        self.assertEqual(data.iloc[0].revenue, 99)
        self.assertEqual(data.iloc[0].revenue_concept, "SalesRevenueNet")

    def test_missing_concepts_and_values(self):
        data, _ = normalize_company_facts(payload({"Revenues": [fact()]}))
        self.assertTrue(pd.isna(data.iloc[0].net_income))
        self.assertTrue(pd.isna(data.iloc[0].free_cash_flow))
        self.assertTrue(validate_fundamentals(data)["valid"])

    def test_quarterly_and_ytd_not_mixed(self):
        common = {"end": "2024-07-28", "filed": "2024-08-28", "fy": 2025,
                  "fp": "Q2", "form": "10-Q", "accn": "q2-2025"}
        data, _ = normalize_company_facts(payload({
            "Revenues": [fact(60, start="2024-04-29", **common),
                         fact(110, start="2024-01-29", **common)],
            "NetCashProvidedByUsedInOperatingActivities": [fact(30, start="2024-01-29", **common)],
            "PaymentsToAcquirePropertyPlantAndEquipment": [fact(10, start="2024-01-29", **common)],
        }))
        quarter = data[data.data_period_type == "quarterly"].iloc[0]
        ytd = data[data.data_period_type == "ytd"].iloc[0]
        self.assertEqual(quarter.revenue, 60)
        self.assertTrue(pd.isna(quarter.operating_cash_flow))
        self.assertTrue(pd.isna(quarter.free_cash_flow))
        self.assertEqual(ytd.revenue, 110)
        self.assertEqual(ytd.free_cash_flow, 20)
        self.assertTrue(validate_fundamentals(data)["valid"])

    def test_comparative_facts_do_not_inherit_filing_fiscal_year(self):
        data, report = normalize_company_facts(payload({"Revenues": [
            fact(100), fact(90, start="2023-01-30", end="2024-01-28"),
        ]}))
        self.assertEqual(len(data), 1)
        self.assertEqual(data.iloc[0].period_end, "2025-01-26")
        self.assertEqual(report["comparative_facts_excluded"], 1)

    def test_duplicates_removed_and_processed_duplicates_rejected(self):
        data, report = normalize_company_facts(payload({"Revenues": [fact(), fact()]}))
        self.assertEqual(len(data), 1)
        self.assertEqual(report["duplicate_source_facts_removed"], 1)
        self.assertFalse(validate_fundamentals(pd.concat([data, data]))["valid"])

    def test_conflicting_values_remain_missing_with_issue(self):
        data, report = normalize_company_facts(payload({"Revenues": [fact(100), fact(101)]}))
        self.assertTrue(pd.isna(data.iloc[0].revenue))
        self.assertEqual(len(report["issues"]), 1)

    def test_same_period_different_filings_retained(self):
        data, _ = normalize_company_facts(payload({"Revenues": [
            fact(100), fact(101, form="10-K/A", accn="amendment", filed="2025-03-01"),
        ]}))
        self.assertEqual(len(data), 2)
        self.assertTrue(validate_fundamentals(data)["valid"])

    def test_invalid_payload(self):
        for source in (None, {}, {"cik": 320193, "facts": {"us-gaap": {}}}, {"cik": CIK}):
            with self.assertRaises(ValueError):
                normalize_company_facts(source)

    def test_malformed_source_identified(self):
        for changes in ({"val": "oops"}, {"val": None}, {"val": float("inf")},
                        {"fy": 0}, {"fp": "bad"}, {"end": "bad"},
                        {"filed": "2020-01-01"}, {"accn": ""}):
            source = payload({"Revenues": [fact(), fact(**changes)]})
            data, report = normalize_company_facts(source)
            self.assertEqual(len(report["issues"]), 1)
            self.assertTrue(validate_fundamentals(data)["valid"])

    def test_validation_errors(self):
        data, _ = normalize_company_facts(payload())
        for column, value in (("ticker", "AAPL"), ("fiscal_year", 0),
                              ("fiscal_period", "bad"), ("period_end", "bad"),
                              ("data_period_type", "quarterly"), ("free_cash_flow", 123),
                              ("capital_expenditures", -10), ("revenue", "bad")):
            changed = data.copy().astype({column: object})
            changed.loc[0, column] = value
            self.assertFalse(validate_fundamentals(changed)["valid"], column)
        self.assertFalse(validate_fundamentals(data.drop(columns="ticker"))["valid"])
        self.assertFalse(validate_fundamentals(data.iloc[:0])["valid"])

    def test_unit_filter_and_diluted_eps(self):
        source = payload()
        source["facts"]["us-gaap"]["EarningsPerShareDiluted"] = {
            "units": {"USD/shares": [fact(2.5)], "USD": [fact(999)]}}
        data, _ = normalize_company_facts(source)
        self.assertEqual(data.iloc[0].eps, 2.5)

    def test_quarter_in_annual_filing_is_q4(self):
        data, _ = normalize_company_facts(payload({"Revenues": [
            fact(100), fact(25, start="2024-10-28"),
        ]}))
        quarter = data[data.data_period_type == "quarterly"].iloc[0]
        self.assertEqual(quarter.fiscal_period, "Q4")
        self.assertTrue(validate_fundamentals(data)["valid"])

    def test_request_uses_nvda_and_user_agent(self):
        body = json.dumps(payload()).encode()
        with patch("nasdaq_research.fundamentals.urlopen") as mocked:
            mocked.return_value.__enter__.return_value.read.return_value = body
            self.assertEqual(download_company_facts("Research contact@example.org"), body)
            request = mocked.call_args.args[0]
            self.assertEqual(request.full_url, ENDPOINT)
            self.assertIn("1045810", request.full_url)
            self.assertEqual(request.get_header("User-agent"), "Research contact@example.org")

    def test_pipeline_raw_bytes_and_offline_replay(self):
        body = json.dumps(payload()).encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("nasdaq_research.fundamentals.download_company_facts", return_value=body):
                summary = run_pipeline(root)
            raw = Path(summary["raw_output_path"])
            self.assertEqual(raw.read_bytes(), body)
            first = Path(summary["processed_output_path"]).read_bytes()
            with patch("nasdaq_research.fundamentals.download_company_facts", side_effect=AssertionError("network forbidden")):
                replay = run_pipeline(root, raw_input=raw)
            self.assertEqual(Path(replay["processed_output_path"]).read_bytes(), first)
            loaded = pd.read_csv(replay["processed_output_path"])
            self.assertTrue(validate_fundamentals(loaded)["valid"])
            self.assertTrue((root / "processed" / "NVDA_validation.json").exists())


if __name__ == "__main__":
    unittest.main()
