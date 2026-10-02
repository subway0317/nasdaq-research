"""Read-only diagnostics for retained denominators versus archived redisclosures.

Differences can reflect rounding, reclassification, redisclosure or restatement;
this module does not classify the cause or replace processed denominators.
"""

import json
from pathlib import Path

import pandas as pd

from nasdaq_research.dates import parse_dates
from nasdaq_research.fundamental_mapping import CONCEPT_MAPPING
from nasdaq_research.fundamentals import KEY_COLUMNS


def comparative_diagnostic(snapshots, fundamentals, raw_path: Path | None):
    result = {"status": "not_available", "difference_snapshot_count": 0,
              "difference_metric_count": 0, "compared_metric_count": 0,
              "differences": [], "conflicting_latest_values": []}
    if raw_path is None or not raw_path.is_file():
        return result
    facts = json.loads(raw_path.read_text(encoding="utf-8")).get("facts", {}).get("us-gaap", {})
    # Physical period and same concept/unit only; comparisons across taxonomy
    # substitutions would not establish a difference in the same disclosure.
    index = {}
    fields = ("revenue", "gross_profit", "operating_income", "net_income",
              "research_and_development", "operating_cash_flow")
    used_concepts = {(concept, CONCEPT_MAPPING[field][1]) for field in fields
                     for concept in fundamentals.get(f"{field}_concept", pd.Series(dtype=object)).dropna().unique()}
    for concept, unit in sorted(used_concepts):
        for fact in facts.get(concept, {}).get("units", {}).get(unit, []):
            if fact.get("form") not in {"10-K", "10-Q", "10-K/A", "10-Q/A"}:
                continue
            if not {"filed", "end", "val", "accn", "start"}.issubset(fact):
                continue
            dates = parse_dates(pd.Series([fact["start"], fact["end"], fact["filed"]]))
            key = (concept, unit, dates.iloc[0], dates.iloc[1])
            index.setdefault(key, []).append((dates.iloc[2], fact["accn"], fact["form"], fact["val"]))
    sources = {key: row for key, (_, row) in zip(
        fundamentals.loc[:, KEY_COLUMNS].itertuples(index=False, name=None), fundamentals.iterrows())}
    different_snapshots = set()
    for _, snapshot in snapshots[snapshots.yoy_reference_accession.notna()].iterrows():
        reference_key = tuple(snapshot[f"yoy_reference_{c}"] for c in KEY_COLUMNS)
        reference = sources[reference_key]
        cutoff = parse_dates(pd.Series([snapshot.filing_date])).iloc[0]
        period = parse_dates(pd.Series([reference.period_start, reference.period_end]))
        snapshot_key = tuple(snapshot[c] for c in KEY_COLUMNS)
        for field in fields:
            concept = reference.get(f"{field}_concept")
            value = reference[field]
            if pd.isna(concept) or pd.isna(value):
                continue
            unit = CONCEPT_MAPPING[field][1]
            known = [record for record in index.get((concept, unit, *period), []) if record[0] <= cutoff]
            if not known:
                continue
            # Match reference selection's date/amendment/accession convention.
            latest = max(known, key=lambda x: (x[0], x[2].endswith("/A"), x[1]))
            tied = {x[3] for x in known if (x[0], x[1], x[2]) == latest[:3]}
            details = {"snapshot_accession": snapshot.accession,
                       "snapshot_period_start": snapshot.period_start,
                       "snapshot_period_end": snapshot.period_end,
                       "snapshot_data_period_type": snapshot.data_period_type,
                       "current_filing_date": snapshot.filing_date,
                       "reference_accession": reference.accession, "metric": field,
                       "concept": concept, "retained_denominator": float(value),
                       "latest_comparative_filing_date": latest[0].strftime("%Y-%m-%d"),
                       "latest_comparative_accession": latest[1]}
            if len(tied) != 1:
                result["conflicting_latest_values"].append(details)
                continue
            result["compared_metric_count"] += 1
            if value != latest[3]:
                result["differences"].append({**details, "latest_comparative_value": latest[3]})
                different_snapshots.add(snapshot_key)
    result.update(status="completed", difference_snapshot_count=len(different_snapshots),
                  difference_metric_count=len(result["differences"]),
                  policy="Same concept/unit and physical period in local archived raw facts; latest known by current filing date. Diagnostic only; denominators unchanged.")
    return result
