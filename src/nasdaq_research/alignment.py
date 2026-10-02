"""Offline Stage 5: filing-date point-in-time alignment of NVDA observations."""

import argparse
import json
from pathlib import Path

import pandas as pd

from nasdaq_research.config import DATA_DIR, FUNDAMENTALS_DATA_DIR, PROCESSED_DATA_DIR
from nasdaq_research.features import read_processed_history
from nasdaq_research.fundamentals import METADATA_COLUMNS, validate_fundamentals

RESEARCH_DATA_DIR = DATA_DIR / "research"
AGE_COLUMNS = ("days_since_filing", "days_since_effective_date")


def _market(market: pd.DataFrame) -> pd.DataFrame:
    result = market.copy().reset_index(drop=True)
    if "date" not in result or result.empty:
        raise ValueError("Nonempty market dates required")
    dates = pd.to_datetime(result.date, errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Market dates must be unique, nonmissing and ascending")
    if not dates.eq(dates.dt.normalize()).all() or dates.dt.tz is not None:
        raise ValueError("Expected timezone-naive calendar dates")
    if "ticker" in result and not result.ticker.eq("NVDA").all():
        raise ValueError("Market ticker must be NVDA")
    result["ticker"] = "NVDA"
    return result


def _observations(fundamentals: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    quality = validate_fundamentals(fundamentals)
    if not quality["valid"]:
        raise ValueError(f"Invalid fundamentals: {quality['errors']}")
    selected = fundamentals[fundamentals.data_period_type.isin(["quarterly", "annual"])].copy()
    selected["_filed"] = pd.to_datetime(selected.filing_date)
    # No guessed weekday calendar: right insertion excludes the filing day,
    # including weekends/holidays. Dates outside coverage are range-censored.
    sessions = pd.DatetimeIndex(dates)
    positions = sessions.searchsorted(selected._filed, side="right")
    selected["effective_date"] = pd.Series(
        [sessions[p] if p < len(sessions) else pd.NaT for p in positions],
        index=selected.index, dtype="datetime64[ns]",
    )
    selected = selected[selected.effective_date.notna()].copy()
    selected["_quarter"] = selected.data_period_type.eq("quarterly").astype(int)
    selected["_amended"] = selected.form.str.endswith("/A").astype(int)
    # At a common effective date: latest filing, latest reported period, standalone
    # quarter, amendment, latest start, then lexical accession. Never combine rows.
    selected = selected.sort_values([
        "effective_date", "_filed", "period_end", "_quarter", "_amended",
        "period_start", "accession",
    ]).drop_duplicates("effective_date", keep="last")
    selected = selected.drop(columns=["ticker", "_filed", "_quarter", "_amended"])
    rename = {c: f"fundamental_{c}" for c in (*METADATA_COLUMNS, "effective_date") if c != "ticker"}
    return selected.rename(columns=rename).reset_index(drop=True)


def _join(market: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    overlap = set(market) & (set(observations) | set(AGE_COLUMNS))
    if overlap:
        raise ValueError(f"Market/fundamental column collision: {sorted(overlap)}")
    left = market.copy()
    left["_trading_date"] = pd.to_datetime(left.date).astype("datetime64[ns]")
    result = pd.merge_asof(
        left, observations, left_on="_trading_date", right_on="fundamental_effective_date",
        direction="backward", allow_exact_matches=True,
    )
    for target, source in zip(AGE_COLUMNS, ("fundamental_filing_date", "fundamental_effective_date")):
        result[target] = (result._trading_date - pd.to_datetime(result[source])).dt.days.astype("Int64")
    return result.drop(columns="_trading_date")


def build_research_dataset(market: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Keep every market row and attach the latest whole eligible observation.

    effective_date is the first *observed* market session strictly after filing;
    pre-window filings are left-censored to the first session in this dataset.
    """
    checked = _market(market)
    return _join(checked, _observations(fundamentals, pd.to_datetime(checked.date)))


def validate_research_dataset(
    data: pd.DataFrame, market: pd.DataFrame, fundamentals: pd.DataFrame,
) -> dict:
    """Audit timing, row preservation and every source value/provenance cell."""
    errors = []
    try:
        checked = _market(market)
        _market(data)
        expected = _join(checked, _observations(fundamentals, pd.to_datetime(checked.date)))
        pd.testing.assert_frame_equal(data.reset_index(drop=True), expected, check_dtype=False)
    except (ValueError, AssertionError, KeyError, TypeError) as exc:
        errors.append(f"Source alignment or row preservation failed: {exc}")
    try:
        trading = pd.to_datetime(data.date)
        filed = pd.to_datetime(data.fundamental_filing_date)
        effective = pd.to_datetime(data.fundamental_effective_date)
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
    fundamentals = pd.read_csv(fundamentals_path)
    result = build_research_dataset(market, fundamentals)
    report = validate_research_dataset(result, market, fundamentals)
    if not report["valid"]:
        raise ValueError(report["errors"])
    report.update({
        "ticker": "NVDA", "market_input": str(market_dir / "NVDA.csv"),
        "fundamentals_input": str(fundamentals_path),
        "calendar_basis": "Observed market sessions; pre-window effective dates are left-censored",
        "pre_window_observations": int((pd.to_datetime(fundamentals.filing_date) < pd.Timestamp(market.date.iloc[0])).sum()),
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
