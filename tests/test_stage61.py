"""Stage 6.1 adversarial regressions; oracle deliberately avoids production joins."""

from datetime import date
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from nasdaq_research.alignment import build_research_dataset, validate_research_dataset
from nasdaq_research.comparative import comparative_diagnostic
from nasdaq_research.dates import parse_dates
from nasdaq_research.features import build_features
from nasdaq_research.provenance import assert_values_equal, csv_with_source_cells
from nasdaq_research.research_features import (
    BALANCE_FIELDS, FEATURE_COLUMNS, build_feature_matrix, build_fundamental_features,
    run_pipeline, validate_feature_matrix,
)
from test_research_features import build, market, observation, snapshot


def mixed_rows():
    return [observation(fy=2026, revenue=100, filing_date="2025-08-27", accession="original"),
            observation(fy=2026, revenue=400, filing_date="2026-09-01", accession="future", form="10-Q/A"),
            observation(revenue=200, filing_date="20260826", accession="current")]


def oracle_states(rows, dates):
    """Small exhaustive oracle: date.fromisoformat, loops and max, no joins/helpers.

    Market rows are observed sessions. A filing is known iff it strictly
    precedes that session. Each stream chooses one complete source observation.
    """
    answer = []
    for trading in dates:
        trading = date.fromisoformat(str(trading))
        states = {}
        for prefix in ("q", "fy", "bs"):
            eligible = []
            for row in rows:
                if date.fromisoformat(row["filing_date"]) >= trading:
                    continue
                basis = row["data_period_type"]
                if prefix == "q" and basis != "quarterly":
                    continue
                if prefix == "fy" and basis != "annual":
                    continue
                if prefix == "bs" and (basis not in ("quarterly", "annual")
                                        or not any(pd.notna(row[c]) for c in BALANCE_FIELDS)):
                    continue
                eligible.append(row)
            if not eligible:
                states[prefix] = None
                continue
            row = max(eligible, key=lambda r: (
                date.fromisoformat(r["filing_date"]), date.fromisoformat(r["period_end"]),
                r["data_period_type"] == "quarterly", r["form"].endswith("/A"),
                date.fromisoformat(r["period_start"]), r["accession"]))
            effective = next(date.fromisoformat(str(d)) for d in dates
                             if date.fromisoformat(str(d)) > date.fromisoformat(row["filing_date"]))
            states[prefix] = (row, effective)
        answer.append(states)
    return answer


class DateTests(unittest.TestCase):
    def test_mixed_date_snapshot_counterexample(self):
        row = snapshot(mixed_rows(), "current")
        self.assertEqual(row.yoy_reference_accession, "original")
        self.assertEqual(row.filing_date, "2026-08-26")
        self.assertEqual(row.q_revenue_growth_yoy, 1.)

    def test_mixed_date_full_pipeline_counterexample(self):
        f, r, _, _ = build(mixed_rows())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fpath = root / "f.csv"; rpath = root / "r.csv"
            f.to_csv(fpath, index=False); r.to_csv(rpath, index=False)
            summary = run_pipeline(fpath, rpath, root, raw_path=None)
            daily = pd.read_csv(root / "NVDA_features.csv")
            self.assertEqual(summary["reference_cutoff_violation_count"], 0)
            self.assertEqual(summary["date_normalization"]["normalized_input_values"]["filing_date"], 1)
            before_amendment = daily[daily.date.eq("2026-08-27")].iloc[0]
            self.assertEqual(before_amendment.q_yoy_reference_accession, "original")
            self.assertEqual(before_amendment.q_revenue_growth_yoy, 1.)

    def test_entire_basic_date_csv_columns_remain_strings(self):
        f, r, _, _ = build([observation()])
        for c in ("filing_date", "period_start", "period_end"):
            f[c] = f[c].str.replace("-", "", regex=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); f.to_csv(root / "f.csv", index=False); r.to_csv(root / "r.csv", index=False)
            self.assertTrue(run_pipeline(root / "f.csv", root / "r.csv", root, None)["validation"]["valid"])

    def test_malformed_and_ambiguous_dates_rejected(self):
        for value in ("2026-02-30", "08/26/2026", "2026-08-26T12:00:00", "9999-12-31", "garbage", None, 20260826):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Invalid filing_date"):
                build_fundamental_features(pd.DataFrame([observation(filing_date=value)]))

    def test_mixed_date_sort_is_chronological_and_shuffle_stable(self):
        rows = mixed_rows(); expected = build_fundamental_features(pd.DataFrame(rows))
        self.assertEqual(expected.accession.tolist(), ["original", "current", "future"])
        pd.testing.assert_frame_equal(expected, build_fundamental_features(pd.DataFrame(rows).iloc[::-1]))

    def test_explicit_cutoff_rejects_bad_reference_selection(self):
        f = pd.DataFrame(mixed_rows())
        future = pd.Series(mixed_rows()[1]); future.filing_date = pd.Timestamp(future.filing_date)
        with patch("nasdaq_research.research_features._reference", return_value=(future, "available")):
            with self.assertRaisesRegex(ValueError, "later than current"):
                build_fundamental_features(f)

    def test_age_columns_are_not_parsed_as_dates(self):
        frame = pd.DataFrame({"q_days_since_effective_date": [0, 2], "days_since_filing": [1, 3]})
        assert_values_equal(frame, frame)

    def test_market_and_state_mixed_dates_preserve_values(self):
        f, research, _, _ = build([observation()])
        research["date"] = research.date.astype(str)
        research.loc[3, "date"] = "20260827"
        research.loc[3, "fundamental_filing_date"] = "20260826"
        s, daily = build_feature_matrix(research, f)
        pd.testing.assert_frame_equal(daily[research.columns], research)
        self.assertEqual(daily.q_accession.iloc[3], f.accession.iloc[0])
        self.assertTrue(validate_feature_matrix(daily, s, research, f)["valid"])

    def test_date_objects_basic_extended_and_timezone_policy(self):
        values = pd.Series(["20260826", "2026-08-26", date(2026, 8, 26), pd.Timestamp("2026-08-26")])
        self.assertTrue(parse_dates(values).eq(pd.Timestamp("2026-08-26")).all())
        for value in (pd.Timestamp("2026-08-26", tz="UTC"), pd.Timestamp("2026-08-26 01:00")):
            with self.assertRaises(ValueError):
                parse_dates(pd.Series([value]))


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.f, self.r, self.s, self.d = build([observation(operating_income=28_440_000_000.)])

    def test_142200_dollar_mutation_rejected_by_both_stages(self):
        bad = self.r.copy(); bad.loc[3, "operating_income"] += 142200
        self.assertFalse(validate_research_dataset(bad, market(), self.f)["valid"])
        with self.assertRaisesRegex(ValueError, "Stage 5 source"):
            build_feature_matrix(bad, self.f)
        bad_daily = self.d.copy(); bad_daily.loc[3, "operating_income"] += 142200
        self.assertFalse(validate_feature_matrix(bad_daily, self.s, self.r, self.f)["valid"])

    def test_one_dollar_cent_and_one_ulp_mutations_rejected(self):
        original = self.r.operating_income.iloc[3]
        for value in (original + 1., original + .01, np.nextafter(original, np.inf)):
            with self.subTest(value=value):
                bad = self.r.copy(); bad.loc[3, "operating_income"] = value
                self.assertFalse(validate_research_dataset(bad, market(), self.f)["valid"])
                bad_daily = self.d.copy(); bad_daily.loc[3, "operating_income"] = value
                self.assertFalse(validate_feature_matrix(bad_daily, self.s, self.r, self.f)["valid"])

    def test_numeric_dtypes_and_paired_nans_allowed(self):
        actual = pd.DataFrame({"amount": pd.Series([1., np.nan, 3.])})
        expected = pd.DataFrame({"amount": pd.Series([1, None, 3], dtype="Int64")})
        assert_values_equal(actual, expected)
        changed = self.r.copy(); changed["operating_income"] = changed.operating_income.astype("Int64")
        self.assertTrue(validate_research_dataset(changed, market(), self.f)["valid"])

    def test_nan_to_zero_or_missing_value_rejected(self):
        for column, value in (("eps", 0.), ("operating_income", np.nan)):
            bad = self.r.copy(); bad.loc[3, column] = value
            self.assertFalse(validate_research_dataset(bad, market(), self.f)["valid"])

    def test_metadata_tampering_rejected(self):
        for column, value in (("q_accession", "wrong"), ("fy_fiscal_year", 2026),
                              ("q_period_end", "2026-07-30"), ("bs_form", "10-K")):
            bad = self.d.copy(); bad.loc[3, column] = value
            self.assertFalse(validate_feature_matrix(bad, self.s, self.r, self.f)["valid"], column)

    def test_snapshot_raw_values_exact_and_ratio_tolerance_explicit(self):
        bad = self.s.copy(); bad.loc[0, "operating_income"] += .01
        self.assertFalse(validate_feature_matrix(self.d, bad, self.r, self.f)["valid"])
        bad = self.d.copy(); bad.loc[3, "q_gross_margin"] += 1e-5
        self.assertFalse(validate_feature_matrix(bad, self.s, self.r, self.f)["valid"])
        good = self.d.copy(); good.loc[3, "q_gross_margin"] += 1e-15
        self.assertTrue(validate_feature_matrix(good, self.s, self.r, self.f)["valid"])

    def test_market_raw_price_mutation_exact(self):
        bad = self.r.copy(); bad.loc[3, "close"] = np.nextafter(bad.close.iloc[3], np.inf)
        self.assertFalse(validate_research_dataset(bad, market(), self.f)["valid"])

    def test_csv_preserves_decimal_tokens_and_canonicalizes_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.csv"
            path.write_text('date,amount,identifier\n20260826,184.01800231933595,"quoted,value"\n')
            data = pd.read_csv(path, dtype={"date": "string"}).assign(extra=.6)
            data["date"] = "2026-08-26"
            exported = csv_with_source_cells(data, path)
            self.assertIn('2026-08-26,184.01800231933595,"quoted,value",0.6', exported)
            loaded = pd.read_csv(StringIO(exported))
            assert_values_equal(loaded, data)
            data.loc[0, "amount"] += .01
            with self.assertRaises(AssertionError):
                csv_with_source_cells(data, path)

    def test_pipeline_reloaded_csv_passes_strict_source_validation(self):
        m = market(); m[["open", "high", "low", "close"]] += 84.01800231933595
        m = build_features(m[["date", "open", "high", "low", "close", "volume"]])
        f, research, _, _ = build([observation()], m)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); f.to_csv(root / "f.csv", index=False); research.to_csv(root / "r.csv", index=False)
            report = run_pipeline(root / "f.csv", root / "r.csv", root, None)
            source = pd.read_csv(root / "r.csv")
            daily = pd.read_csv(root / "NVDA_features.csv")
            s = pd.read_csv(root / "NVDA_fundamental_features.csv")
            assert_values_equal(daily[source.columns], source)
            self.assertTrue(validate_feature_matrix(daily, s, source, f)["valid"])
            self.assertTrue(report["validation"]["checks"]["csv_roundtrip_preserves_source_values"])


class StateTests(unittest.TestCase):
    def test_independent_exhaustive_point_in_time_oracle(self):
        dates = ["2026-08-24", "2026-08-26", "2026-08-27", "2026-08-28", "2026-09-01", "2026-09-02", "2026-09-03"]
        rows = [observation(fy=2026, fp="FY", basis="annual", accession="annual"),
                observation(fp="Q1", accession="pre-sample"),
                observation(accession="same-a", filing_date="2026-08-26", revenue=100),
                observation(accession="same-z", filing_date="2026-08-26", revenue=300, form="10-Q/A"),
                observation(accession="weekend", filing_date="2026-08-29", revenue=400, form="10-Q/A"),
                observation(fy=2026, fp="FY", basis="annual", accession="annual-amend", filing_date="2026-09-01", form="10-K/A", revenue=800),
                observation(accession="future", filing_date="2026-09-04", revenue=900)]
        expected = oracle_states(rows, dates)
        _, _, _, daily = build(rows, market(dates))
        for i, states in enumerate(expected):
            for prefix, (source, effective) in states.items():
                self.assertEqual(daily[f"{prefix}_accession"].iloc[i], source["accession"])
                self.assertEqual(daily[f"{prefix}_period_end"].iloc[i], source["period_end"])
                self.assertEqual(daily[f"{prefix}_effective_date"].iloc[i], pd.Timestamp(effective))
                self.assertEqual(daily[f"{prefix}_days_since_filing"].iloc[i],
                                 (date.fromisoformat(dates[i]) - date.fromisoformat(source["filing_date"])).days)
                if prefix == "bs":
                    for field in BALANCE_FIELDS:
                        got, want = daily[f"bs_{field}"].iloc[i], source[field]
                        self.assertTrue((pd.isna(got) and pd.isna(want)) or got == want)
                    self.assertEqual(daily.current_ratio.iloc[i], source["current_assets"] / source["current_liabilities"])
                else:
                    self.assertEqual(daily[f"{prefix}_gross_margin"].iloc[i], source["gross_profit"] / source["revenue"])
        self.assertEqual(daily.q_accession.tolist(), ["pre-sample", "pre-sample", "same-z", "same-z", "weekend", "weekend", "weekend"])
        self.assertEqual(daily.fy_accession.iloc[-1], "annual-amend")

    def test_quarter_update_preserves_annual_and_annual_update_preserves_quarter(self):
        rows = [observation(fy=2026, fp="FY", basis="annual", revenue=500, accession="fy-old"),
                observation(accession="q-new"),
                observation(fy=2026, fp="FY", basis="annual", revenue=800, accession="fy-new", filing_date="2026-09-01")]
        _, _, _, d = build(rows)
        self.assertEqual(d.fy_accession.iloc[3], "fy-old")
        self.assertEqual(d.q_accession.iloc[3], "q-new")
        self.assertEqual(d.fy_accession.iloc[-1], "fy-new")
        self.assertEqual(d.q_accession.iloc[-1], "q-new")
        self.assertTrue(d.fy_gross_margin.notna().all())

    def test_early_only_quarter_and_only_annual_states(self):
        for row, present, missing in ((observation(), "q", "fy"), (observation(fy=2026, fp="FY", basis="annual"), "fy", "q")):
            _, _, _, d = build([row])
            self.assertTrue(d[f"{present}_accession"].iloc[-1] == row["accession"])
            self.assertTrue(d[f"{missing}_accession"].isna().all())
        _, _, _, d = build([observation(), observation(fy=2026, fp="FY", basis="annual")])
        self.assertTrue((d.q_accession.notna() & d.fy_accession.notna()).iloc[3:].all())

    def test_future_q_fy_and_amendments_do_not_change_prefix(self):
        for fp, basis in (("Q2", "quarterly"), ("FY", "annual")):
            old = observation(fy=2026, fp=fp, basis=basis, accession="old")
            future = observation(fy=2026, fp=fp, basis=basis, accession="future", filing_date="2026-09-01",
                                 form="10-Q/A" if basis == "quarterly" else "10-K/A")
            _, _, _, baseline = build([old, future])
            future["revenue"] = 999999
            _, _, _, changed = build([old, future])
            pd.testing.assert_frame_equal(baseline.iloc[:7], changed.iloc[:7])
            self.assertEqual(changed[f"{'q' if basis == 'quarterly' else 'fy'}_accession"].iloc[-1], "future")

    def test_late_older_period_amendments_move_each_state_backward(self):
        for fp, basis in (("Q2", "quarterly"), ("FY", "annual")):
            newer = observation(fy=2026 if basis == "annual" else 2027, fp=fp, basis=basis, accession="newer")
            older = observation(fy=int(newer["fiscal_year"]) - 1, fp=fp, basis=basis, accession="old-amend",
                                filing_date="2026-09-01", form="10-K/A" if basis == "annual" else "10-Q/A")
            _, _, _, d = build([newer, older])
            prefix = "fy" if basis == "annual" else "q"
            self.assertEqual(d[f"{prefix}_accession"].iloc[6], "newer")
            self.assertEqual(d[f"{prefix}_accession"].iloc[7], "old-amend")
            self.assertLess(d[f"{prefix}_period_end"].iloc[7], d[f"{prefix}_period_end"].iloc[6])

    def test_balance_latest_snapshot_and_no_metric_fill(self):
        old = observation(fy=2026, fp="FY", basis="annual", accession="bs-annual")
        new = observation(accession="bs-quarter", long_term_debt=np.nan, current_assets=360.)
        _, _, _, d = build([old, new])
        self.assertEqual(d.bs_accession.iloc[0], "bs-annual")
        self.assertEqual(d.bs_accession.iloc[3], "bs-quarter")
        self.assertEqual(d.current_ratio.iloc[3], 4.)
        self.assertTrue(pd.isna(d.bs_long_term_debt.iloc[3]))
        self.assertTrue(pd.isna(d.debt_to_assets.iloc[3]))

    def test_income_only_snapshot_keeps_whole_older_bs_state(self):
        old = observation(fy=2026, fp="FY", basis="annual", accession="bs-annual")
        new = observation(accession="income-only", **{c: np.nan for c in BALANCE_FIELDS})
        _, _, _, d = build([old, new])
        self.assertEqual(d.q_accession.iloc[-1], "income-only")
        self.assertTrue(d.bs_accession.eq("bs-annual").all())
        self.assertTrue(d.current_ratio.eq(2.).all())

    def test_state_first_date_filing_day_weekend_and_missing_session(self):
        dates = ["2026-08-26", "2026-08-27", "2026-08-28", "2026-09-02", "2026-09-03"]
        for filed, index in (("2026-08-26", 1), ("2026-08-29", 3), ("2026-09-01", 3)):
            _, _, _, d = build([observation(filing_date=filed)], market(dates))
            self.assertTrue(d.q_accession.iloc[:index].isna().all())
            self.assertEqual(d.q_effective_date.iloc[index], pd.Timestamp(dates[index]))

    def test_pre_sample_flags_and_actual_filing_age(self):
        annual = observation(fy=2026, fp="FY", basis="annual")
        _, _, s, d = build([annual, observation()])
        self.assertTrue(d.fy_effective_date_is_sample_truncated.all())
        self.assertEqual(d.fy_effective_date_basis.iloc[0], "sample_start_truncated")
        self.assertEqual(d.fy_days_since_effective_date.iloc[0], 0)
        self.assertEqual(d.fy_days_since_filing.iloc[0], 180)
        self.assertEqual(d.q_effective_date_basis.iloc[3], "observed_next_session")
        self.assertTrue(s.effective_date_is_sample_truncated.iloc[0])

    def test_shuffle_ties_and_multiple_filings_on_same_effective_date(self):
        rows = [observation(accession="a", filing_date="2026-08-28"),
                observation(accession="b", filing_date="2026-08-29"),
                observation(accession="z", filing_date="2026-08-29", form="10-Q/A"),
                observation(fy=2026, fp="FY", basis="annual", accession="annual", filing_date="2026-08-29")]
        _, _, expected_s, expected_d = build(rows)
        self.assertEqual(expected_d.q_accession.iloc[5], "z")
        self.assertEqual(expected_d.fy_accession.iloc[5], "annual")
        self.assertEqual(expected_d.bs_accession.iloc[5], "z")
        for seed in range(3):
            shuffled = pd.DataFrame(rows).sample(frac=1, random_state=seed).to_dict("records")
            _, _, s, d = build(shuffled)
            pd.testing.assert_frame_equal(expected_s, s); pd.testing.assert_frame_equal(expected_d, d)

    def test_ytd_cannot_update_any_state(self):
        old = observation(fp="Q1", accession="standalone")
        ytd = observation(basis="ytd", period_start="2026-02-01", accession="ytd", revenue=999)
        _, _, s, d = build([old, ytd])
        self.assertEqual(len(s), 1)
        self.assertTrue(d.q_accession.eq("standalone").all())
        self.assertTrue(d.bs_accession.eq("standalone").all())

    def test_future_market_values_do_not_change_past_states_or_features(self):
        m = market(); rows = [observation(), observation(fy=2026, fp="FY", basis="annual")]
        _, _, _, expected = build(rows, m)
        raw = m[["date", "open", "high", "low", "close", "volume"]].copy()
        raw.loc[8:, ["open", "high", "low", "close", "volume"]] *= 2
        _, _, _, actual = build(rows, build_features(raw))
        pd.testing.assert_frame_equal(expected.iloc[:8], actual.iloc[:8])


class CollisionTests(unittest.TestCase):
    def test_multiple_internal_market_columns_preserved(self):
        m = market().assign(_trading_date="user-value", __trading_date=np.arange(10), _filed=42)
        f, r, s, d = build([observation()], m)
        pd.testing.assert_frame_equal(r[m.columns], m)
        pd.testing.assert_frame_equal(d[m.columns], m)
        self.assertTrue(validate_research_dataset(r, m, f)["valid"])
        self.assertTrue(validate_feature_matrix(d, s, r, f)["valid"])

    def test_fundamental_internal_key_collision_rejected_clearly(self):
        for name in ("_filed", "_period_start", "_period_end", "_quarter", "_amended", "effective_date"):
            f = pd.DataFrame([observation(**{name: "user-value"})])
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Reserved"):
                build_research_dataset(market(), f)


class PeriodAndVintageTests(unittest.TestCase):
    def test_364_and_371_day_gaps_accept_52_and_53_week_years(self):
        for gap in (364, 371):
            for fp, basis in (("Q2", "quarterly"), ("FY", "annual")):
                current = observation(fp=fp, basis=basis)
                if basis == "annual":
                    current["period_end"] = "2027-01-30"  # 52-week current year; no shared boundary day
                previous = observation(fy=2026, fp=fp, basis=basis, revenue=100,
                    period_start=(pd.Timestamp(current["period_start"]) - pd.Timedelta(days=gap)).strftime("%Y-%m-%d"),
                    period_end=(pd.Timestamp(current["period_end"]) - pd.Timedelta(days=gap)).strftime("%Y-%m-%d"))
                row = snapshot([previous, current])
                self.assertEqual(row.yoy_comparison_status, "available")
                self.assertEqual(row[f"{'q' if basis == 'quarterly' else 'fy'}_revenue_growth_yoy"], 1.)

    def test_period_gap_boundaries_329_330_400_401(self):
        current = observation()
        for gap, valid in ((329, False), (330, True), (400, True), (401, False)):
            previous = observation(fy=2026, revenue=100,
                period_start=(pd.Timestamp(current["period_start"]) - pd.Timedelta(days=gap)).strftime("%Y-%m-%d"),
                period_end=(pd.Timestamp(current["period_end"]) - pd.Timedelta(days=gap)).strftime("%Y-%m-%d"))
            previous["filing_date"] = (pd.Timestamp(previous["period_end"]) + pd.Timedelta(days=20)).strftime("%Y-%m-%d")
            row = snapshot([previous, current])
            self.assertEqual(row.yoy_comparison_status, "available" if valid else "noncomparable_period_dates")
            self.assertEqual(pd.notna(row.q_revenue_growth_yoy), valid)

    def test_fiscal_calendar_change_with_matching_labels_is_conservative(self):
        old = observation(fy=2026, filing_date="2026-08-01", period_start="2026-02-01", period_end="2026-04-30")
        row = snapshot([old, observation()])
        self.assertEqual(row.yoy_comparison_status, "noncomparable_period_dates")
        self.assertTrue(pd.isna(row.q_revenue_growth_yoy))

    def test_previous_fiscal_key_with_two_physical_periods_is_ambiguous(self):
        old = observation(fy=2026, accession="old")
        other = observation(fy=2026, accession="duplicate-key", period_start="2025-05-02")
        row = snapshot([old, other, observation()])
        self.assertEqual(row.yoy_comparison_status, "ambiguous_previous_period")
        self.assertTrue(pd.isna(row.q_revenue_growth_yoy))

    def test_latest_processed_version_deterministic_and_frozen(self):
        previous = observation(fy=2026, revenue=100, accession="old")
        amend = observation(fy=2026, revenue=150, accession="amend-z", filing_date="2026-08-25", form="10-Q/A")
        tie = {**amend, "accession": "amend-a", "revenue": 50.}
        future = {**amend, "accession": "future", "filing_date": "2026-09-01", "revenue": 999.}
        expected = snapshot([previous, amend, tie, future, observation()], observation()["accession"])
        self.assertEqual(expected.yoy_reference_accession, "amend-z")
        self.assertAlmostEqual(expected.q_revenue_growth_yoy, 200 / 150 - 1)
        pd.testing.assert_series_equal(expected, snapshot([observation(), future, tie, amend, previous], observation()["accession"]))

    def test_raw_comparative_diagnostic_does_not_replace_processed_denominator(self):
        previous = observation(fy=2026, revenue=100, accession="old")
        current = observation(accession="current")
        f = pd.DataFrame([previous, current]); s = build_fundamental_features(f)
        def fact(value, filed, accession):
            return {"start": previous["period_start"], "end": previous["period_end"], "filed": filed,
                    "val": value, "form": "10-Q", "accn": accession}
        facts = [fact(100, "2025-08-26", "old"), fact(120, "2026-08-01", "redisclosed"), fact(999, "2026-09-01", "future")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.json"
            path.write_text(json.dumps({"facts": {"us-gaap": {"Revenues": {"units": {"USD": facts}}}}}))
            diagnostic = comparative_diagnostic(s, f, path)
        self.assertEqual(diagnostic["difference_snapshot_count"], 1)
        self.assertEqual(diagnostic["difference_metric_count"], 1)
        self.assertEqual(diagnostic["differences"][0]["latest_comparative_value"], 120)
        self.assertEqual(s.q_revenue_growth_yoy.iloc[-1], 1.)

    def test_absent_local_raw_diagnostic_is_explicit(self):
        f = pd.DataFrame([observation()]); s = build_fundamental_features(f)
        self.assertEqual(comparative_diagnostic(s, f, None)["status"], "not_available")


if __name__ == "__main__":
    unittest.main()
