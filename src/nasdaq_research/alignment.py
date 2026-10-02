"""Offline Stage 5: filing-date point-in-time alignment of NVDA observations."""

import argparse
import json
from pathlib import Path

import pandas as pd

from nasdaq_research.config import DATA_DIR, FUNDAMENTALS_DATA_DIR, PROCESSED_DATA_DIR
from nasdaq_research.features import WARMUP_ROWS, read_processed_history
from nasdaq_research.fundamentals import METADATA_COLUMNS, validate_fundamentals
from nasdaq_research.dates import normalize_fundamental_dates, parse_dates
from nasdaq_research.provenance import assert_values_equal

RESEARCH_DATA_DIR = DATA_DIR / "research"
AGE_COLUMNS = ("days_since_filing", "days_since_effective_date")


def _market(market: pd.DataFrame) -> pd.DataFrame:
    result = market.copy().reset_index(drop=True)
    if "date" not in result or result.empty:
        raise ValueError("Nonempty market dates required")
    dates = parse_dates(result.date)
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Market dates must be unique, nonmissing and ascending")
    if not dates.eq(dates.dt.normalize()).all() or dates.dt.tz is not None:
        raise ValueError("Expected timezone-naive calendar dates")
    if "ticker" in result and not result.ticker.eq("NVDA").all():
        raise ValueError("Market ticker must be NVDA")
    result["ticker"] = "NVDA"
    return result


def effective_dates(filing_dates: pd.Series, dates: pd.Series) -> pd.Series:
    """First observed session strictly after each filing, with coverage censoring."""
    sessions = pd.DatetimeIndex(parse_dates(dates))
    if sessions.has_duplicates or not sessions.is_monotonic_increasing:
        raise ValueError("Observed sessions must be unique and ascending")
    positions = sessions.searchsorted(parse_dates(filing_dates), side="right")
    return pd.Series(
        [sessions[p] if p < len(sessions) else pd.NaT for p in positions],
        index=filing_dates.index, dtype="datetime64[ns]",
    )


def _observations(
    fundamentals: pd.DataFrame, dates: pd.Series, *, allow_empty: bool = False,
) -> pd.DataFrame:
    fundamentals = normalize_fundamental_dates(fundamentals)
    quality = validate_fundamentals(fundamentals)
    if not quality["valid"] and not (allow_empty and quality["errors"] == ["No observations"]):
        raise ValueError(f"Invalid fundamentals: {quality['errors']}")
    internal = {"_filed", "_period_start", "_period_end", "_quarter", "_amended"}
    reserved = internal | {"effective_date"}
    if reserved & set(fundamentals):
        raise ValueError(f"Reserved fundamental columns: {sorted(reserved & set(fundamentals))}")
    selected = fundamentals[fundamentals.data_period_type.isin(["quarterly", "annual"])].copy()
    selected["_filed"] = parse_dates(selected.filing_date)
    selected["_period_start"] = parse_dates(selected.period_start)
    selected["_period_end"] = parse_dates(selected.period_end)
    # No guessed weekday calendar: right insertion excludes the filing day,
    # including weekends/holidays. Dates outside coverage are range-censored.
    selected["effective_date"] = effective_dates(selected._filed, dates)
    selected = selected[selected.effective_date.notna()].copy()
    selected["_quarter"] = selected.data_period_type.eq("quarterly").astype(int)
    selected["_amended"] = selected.form.str.endswith("/A").astype(int)
    # At a common effective date: latest filing, latest reported period, standalone
    # quarter, amendment, latest start, then lexical accession. Never combine rows.
    selected = selected.sort_values([
        "effective_date", "_filed", "_period_end", "_quarter", "_amended",
        "_period_start", "accession",
    ], kind="stable").drop_duplicates("effective_date", keep="last")
    selected = selected.drop(columns=["ticker", *sorted(internal)])
    rename = {c: f"fundamental_{c}" for c in (*METADATA_COLUMNS, "effective_date") if c != "ticker"}
    return selected.rename(columns=rename).reset_index(drop=True)


def _join(market: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    overlap = set(market) & (set(observations) | set(AGE_COLUMNS))
    if overlap:
        raise ValueError(f"Market/fundamental column collision: {sorted(overlap)}")
    left = market.copy()
    key = "_trading_date"
    while key in set(left) | set(observations):
        key = "_" + key
    left[key] = parse_dates(left.date)
    result = pd.merge_asof(
        left, observations, left_on=key, right_on="fundamental_effective_date",
        direction="backward", allow_exact_matches=True,
    )
    for target, source in zip(AGE_COLUMNS, ("fundamental_filing_date", "fundamental_effective_date")):
        result[target] = (result[key] - parse_dates(result[source], allow_missing=True)).dt.days.astype("Int64")
    result = result.drop(columns=key)
    # Independent of the fundamental selection: no user market field may vanish,
    # move or change. This also protects all dynamically chosen temporary names.
    assert_values_equal(result.loc[:, market.columns], market)
    return result


def build_research_dataset(
    market: pd.DataFrame, fundamentals: pd.DataFrame, *, allow_empty: bool = False,
) -> pd.DataFrame:
    """Keep every market row and attach the latest whole eligible observation.

    effective_date is the first *observed* market session strictly after filing;
    pre-window filings are left-censored to the first session in this dataset.
    """
    checked = _market(market)
    return _join(checked, _observations(fundamentals, parse_dates(checked.date), allow_empty=allow_empty))


def validate_research_dataset(
    data: pd.DataFrame, market: pd.DataFrame, fundamentals: pd.DataFrame,
) -> dict:
    """Audit timing, row preservation and every source value/provenance cell."""
    errors = []
    try:
        checked = _market(market)
        _market(data)
        assert_values_equal(data.loc[:, checked.columns], checked)
        expected = _join(checked, _observations(fundamentals, parse_dates(checked.date)))
        # Raw SEC values and metadata are exact. Only inherited Stage 3 derived
        # arithmetic can differ by at most DERIVED_ATOL after CSV serialization.
        assert_values_equal(data, expected, derived_columns=WARMUP_ROWS)
    except (ValueError, AssertionError, KeyError, TypeError) as exc:
        errors.append(f"Source alignment or row preservation failed: {exc}")
    try:
        trading = parse_dates(data.date)
        filed = parse_dates(data.fundamental_filing_date, allow_missing=True)
        effective = parse_dates(data.fundamental_effective_date, allow_missing=True)
        if (filed >= trading).any():
            errors.append("Filing must strictly precede trading date")
        if (effective > trading).any():
            errors.append("Future effective date")
        if (data.days_since_filing.dropna() < 0).any():
            errors.append("Negative information age")
    except (AttributeError, ValueError, TypeError) as exc:
        errors.append(f"Timing validation failed: {exc}")
    return {"valid": not errors, "errors": errors, "row_count": len(data),
            "unmatched_rows": int(data.fundamental_accession.isna().sum()) if "fundamental_accession" in data else None}


def run_pipeline(
    market_dir: Path = PROCESSED_DATA_DIR,
    fundamentals_path: Path = FUNDAMENTALS_DATA_DIR / "processed" / "NVDA_fundamentals.csv",
    output_dir: Path = RESEARCH_DATA_DIR,
) -> dict:
    """Read existing Stage 3/4 artifacts only; save dataset and audit report."""
    market = read_processed_history("NVDA", market_dir)
    fundamentals = pd.read_csv(fundamentals_path, dtype={c: "string" for c in ("filing_date", "period_start", "period_end")})
    result = build_research_dataset(market, fundamentals)
    report = validate_research_dataset(result, market, fundamentals)
    if not report["valid"]:
        raise ValueError(report["errors"])
    report.update({
        "ticker": "NVDA", "market_input": str(market_dir / "NVDA.csv"),
        "fundamentals_input": str(fundamentals_path),
        "calendar_basis": "Observed market sessions; pre-window effective dates are left-censored",
        "pre_window_observations": int((parse_dates(fundamentals.filing_date) < parse_dates(market.date).iloc[0]).sum()),
        "excluded_ytd": int(fundamentals.data_period_type.eq("ytd").sum()),
        "amended_observations": int(fundamentals.form.str.endswith("/A").sum()),
        "repeated_period_observations": int(fundamentals.duplicated(["period_end", "data_period_type"], keep=False).sum()),
        "matched_accessions": result.fundamental_accession.dropna().unique().tolist(),
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "NVDA_research.csv", index=False)
    (output_dir / "NVDA_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--fundamentals-path", type=Path, default=FUNDAMENTALS_DATA_DIR / "processed" / "NVDA_fundamentals.csv")
    parser.add_argument("--output-dir", type=Path, default=RESEARCH_DATA_DIR)
    args = parser.parse_args()
    print(json.dumps(run_pipeline(args.market_dir, args.fundamentals_path, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
