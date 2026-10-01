"""Reproducible NVDA Company Facts pipeline, with explicit cumulative periods."""

import argparse
from collections import defaultdict
from datetime import date
import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd

from nasdaq_research.config import FUNDAMENTALS_DATA_DIR
from nasdaq_research.fundamental_mapping import CONCEPT_MAPPING, FINANCIAL_FIELDS

CIK = 1045810
ENDPOINT = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK:010d}.json"
DEFAULT_USER_AGENT = "NasdaqResearch/0.1 (individual academic research; NVDA fundamentals)"
KEY_COLUMNS = ("accession", "period_start", "period_end", "data_period_type")
METADATA_COLUMNS = (
    "ticker", "fiscal_year", "fiscal_period", "period_start", "period_end",
    "filing_date", "form", "data_period_type", "accession",
)


def download_company_facts(user_agent: str = DEFAULT_USER_AGENT) -> bytes:
    """Fetch the official NVDA response; retain its original bytes for archival."""
    if not user_agent.strip():
        raise ValueError("SEC User-Agent must be nonempty")
    request = Request(ENDPOINT, headers={"User-Agent": user_agent, "Accept": "application/json"})
    with urlopen(request, timeout=60) as response:
        body = response.read()
    _check_payload(json.loads(body))
    return body


def _check_payload(payload: dict[str, Any]) -> None:
    """Reject another company's or malformed Company Facts response."""
    if not isinstance(payload, dict) or str(payload.get("cik", "")).lstrip("0") != str(CIK):
        raise ValueError("Expected NVDA CIK 0001045810")
    facts = payload.get("facts")
    if not isinstance(facts, dict) or not isinstance(facts.get("us-gaap"), dict):
        raise ValueError("Missing US-GAAP facts")


def _date(value: Any) -> date:
    """Require an ISO calendar date."""
    return date.fromisoformat(value)


def normalize_company_facts(payload: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map current filing periods, preserving YTD and accession-level provenance.

    Only each accession's latest reported end is used: comparative facts carry
    the reporting filing's fy/fp, not their own historical fiscal metadata.
    No cross-filing joins or cumulative subtraction are performed.
    """
    _check_payload(payload)
    facts = payload["facts"]["us-gaap"]
    observations = []
    issues: list[dict[str, Any]] = []
    for field, (concepts, unit, basis) in CONCEPT_MAPPING.items():
        for priority, concept in enumerate(concepts):
            for fact in facts.get(concept, {}).get("units", {}).get(unit, []):
                if fact.get("form") not in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                    continue
                try:
                    end, filed = _date(fact["end"]), _date(fact["filed"])
                    start = _date(fact["start"]) if basis == "duration" else None
                    fy = fact["fy"]
                    if isinstance(fy, bool) or not isinstance(fy, int) or not 1990 <= fy <= 2100:
                        raise ValueError("Invalid fiscal year")
                    if fact["fp"] not in ("FY", "Q1", "Q2", "Q3", "Q4"):
                        raise ValueError("Invalid fiscal period")
                    if not isinstance(fact["accn"], str) or not fact["accn"]:
                        raise ValueError("Missing accession")
                    value = fact["val"]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError("Non-finite/nonnumeric financial value")
                    if filed < end or (start is not None and start > end):
                        raise ValueError("Invalid date chronology")
                    observations.append({"field": field, "concept": concept, "priority": priority,
                                         "basis": basis, "start": start, "end": end,
                                         "filed": filed, "fy": fy, "fp": fact["fp"],
                                         "form": fact["form"], "accn": fact["accn"], "val": value})
                except (KeyError, TypeError, ValueError) as exc:
                    issues.append({"field": field, "concept": concept, "reason": str(exc)})
    latest: dict[str, date] = {}
    for obs in observations:
        latest[obs["accn"]] = max(latest.get(obs["accn"], obs["end"]), obs["end"])
    duration_groups: dict[tuple, list] = defaultdict(list)
    instants: dict[tuple, list] = defaultdict(list)
    for obs in observations:
        if obs["end"] != latest[obs["accn"]]:
            continue
        if obs["basis"] == "instant":
            instants[(obs["accn"], obs["end"])].append(obs)
        else:
            days = (obs["end"] - obs["start"]).days + 1
            annual_form = obs["form"].startswith("10-K")
            if 330 <= days <= 400 and annual_form and obs["fp"] == "FY":
                period_type = "annual"
            elif 70 <= days <= 110:
                period_type = "quarterly"
            elif 111 <= days <= 300 and not annual_form and obs["fp"] in ("Q2", "Q3"):
                period_type = "ytd"
            else:
                issues.append({"field": obs["field"], "reason": f"Unsupported duration {days} days"})
                continue
            duration_groups[(obs["accn"], obs["start"], obs["end"], period_type)].append(obs)
    rows = []
    duplicates = 0
    for (accn, start, end, period_type), items in sorted(duration_groups.items()):
        first = items[0]
        metadata = {(o["fy"], o["fp"], o["filed"], o["form"]) for o in items}
        if len(metadata) != 1:
            issues.append({"accession": accn, "reason": "Conflicting fiscal/filing metadata"})
            continue
        row = dict(zip(METADATA_COLUMNS, (
            "NVDA", first["fy"], "Q4" if period_type == "quarterly" and first["fp"] == "FY" else first["fp"],
            start.isoformat(), end.isoformat(), first["filed"].isoformat(),
            first["form"], period_type, accn,
        )))
        candidates = defaultdict(list)
        for obs in items + instants.get((accn, end), []):
            candidates[obs["field"]].append(obs)
        for field in CONCEPT_MAPPING:
            options = candidates[field]
            row[field] = None
            row[f"{field}_concept"] = None
            if options:
                priority = min(o["priority"] for o in options)
                preferred = [o for o in options if o["priority"] == priority]
                values = {o["val"] for o in preferred}
                duplicates += len(preferred) - len(values)
                if len(values) != 1:
                    issues.append({"accession": accn, "field": field,
                                   "reason": "Conflicting values at preferred concept; left missing"})
                    continue
                row[field] = preferred[0]["val"]
                row[f"{field}_concept"] = preferred[0]["concept"]
        ocf, capex = row["operating_cash_flow"], row["capital_expenditures"]
        row["free_cash_flow"] = ocf - capex if ocf is not None and capex is not None else None
        rows.append(row)
    columns = [*METADATA_COLUMNS, *FINANCIAL_FIELDS,
               *(f"{field}_concept" for field in CONCEPT_MAPPING)]
    result = pd.DataFrame(rows, columns=columns).sort_values(
        ["period_end", "filing_date", "period_start", "accession"], ignore_index=True)
    report = {"issues": issues, "duplicate_source_facts_removed": duplicates,
              "comparative_facts_excluded": sum(o["end"] != latest[o["accn"]] for o in observations)}
    return result, report


def validate_fundamentals(data: pd.DataFrame) -> dict[str, Any]:
    """Check metadata, period basis, numeric values, uniqueness, and FCF identity."""
    errors = []
    required = set(METADATA_COLUMNS) | set(FINANCIAL_FIELDS)
    if required - set(data.columns):
        return {"valid": False, "errors": [f"Missing columns: {sorted(required - set(data.columns))}"]}
    if data.empty:
        errors.append("No observations")
    if data.duplicated(list(KEY_COLUMNS)).any():
        errors.append("Duplicate observation")
    for index, row in data.iterrows():
        try:
            if row.ticker != "NVDA":
                raise ValueError("Ticker must be NVDA")
            if isinstance(row.fiscal_year, bool):
                raise ValueError("Invalid fiscal year")
            fy = float(row.fiscal_year)
            if not math.isfinite(fy) or not fy.is_integer() or not 1990 <= fy <= 2100:
                raise ValueError("Invalid fiscal year")
            if row.fiscal_period not in ("FY", "Q1", "Q2", "Q3", "Q4"):
                raise ValueError("Invalid fiscal period")
            start, end, filed = _date(row.period_start), _date(row.period_end), _date(row.filing_date)
            days = (end - start).days + 1
            if start > end or filed < end or row.form not in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                raise ValueError("Invalid chronology/form")
            valid_period = (
                row.data_period_type == "annual" and 330 <= days <= 400
                and row.form.startswith("10-K") and row.fiscal_period == "FY"
                or row.data_period_type == "quarterly" and 70 <= days <= 110 and row.fiscal_period != "FY"
                or row.data_period_type == "ytd" and 111 <= days <= 300
                and row.form.startswith("10-Q") and row.fiscal_period in ("Q2", "Q3")
            )
            if not valid_period or not isinstance(row.accession, str) or not row.accession:
                raise ValueError("Invalid period type/accession")
            for field in FINANCIAL_FIELDS:
                value = row[field]
                if pd.isna(value):
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Invalid numeric field: {field}")
            if pd.notna(row.capital_expenditures) and row.capital_expenditures < 0:
                raise ValueError("CapEx must be a positive expenditure")
            inputs_present = pd.notna(row.operating_cash_flow) and pd.notna(row.capital_expenditures)
            if inputs_present:
                if pd.isna(row.free_cash_flow) or not math.isclose(
                    row.free_cash_flow, row.operating_cash_flow - row.capital_expenditures,
                    rel_tol=1e-9, abs_tol=0.01,
                ):
                    raise ValueError("FCF does not equal OCF - CapEx")
            elif pd.notna(row.free_cash_flow):
                raise ValueError("FCF present without both source fields")
        except (ValueError, TypeError, OverflowError) as exc:
            errors.append(f"Row {index}: {exc}")
    return {"valid": not errors, "errors": errors,
            "missing_counts": data.loc[:, FINANCIAL_FIELDS].isna().sum().to_dict()}


def run_pipeline(
    output_dir: Path = FUNDAMENTALS_DATA_DIR, user_agent: str = DEFAULT_USER_AGENT,
    raw_input: Path | None = None,
) -> dict[str, Any]:
    """Download/archive or replay raw NVDA facts, validate, and save outputs."""
    raw_path = output_dir / "raw" / "NVDA_companyfacts.json"
    processed_path = output_dir / "processed" / "NVDA_fundamentals.csv"
    body = raw_input.read_bytes() if raw_input is not None else download_company_facts(user_agent)
    payload = json.loads(body)
    _check_payload(payload)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(body)
    data, normalization = normalize_company_facts(payload)
    validation = validate_fundamentals(data)
    if not validation["valid"]:
        raise ValueError(f"Fundamental validation failed: {validation['errors']}")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(processed_path, index=False)
    summary = {
        "ticker": "NVDA", "endpoint": ENDPOINT,
        "annual_observations": int((data.data_period_type == "annual").sum()),
        "quarterly_observations": int((data.data_period_type == "quarterly").sum()),
        "ytd_observations": int((data.data_period_type == "ytd").sum()),
        "date_range": [data.period_end.min(), data.period_end.max()],
        "mapped_fields": [field for field in FINANCIAL_FIELDS if data[field].notna().any()],
        "missing_fields": [field for field in FINANCIAL_FIELDS if data[field].isna().all()],
        "raw_output_path": str(raw_path), "processed_output_path": str(processed_path),
        "validation": validation, "normalization": normalization,
    }
    (processed_path.parent / "NVDA_validation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    """Run a minimal download pipeline or reproduce it from an archived response."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input", type=Path, help="Replay an archived response without network")
    args = parser.parse_args()
    summary = run_pipeline(user_agent=os.getenv("SEC_USER_AGENT", DEFAULT_USER_AGENT), raw_input=args.raw_input)
    print(json.dumps({key: value for key, value in summary.items() if key != "normalization"}, indent=2))
    print(f"Normalization issues: {len(summary['normalization']['issues'])}; details in NVDA_validation.json")


if __name__ == "__main__":
    main()
