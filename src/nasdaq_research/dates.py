"""Explicit calendar-date parsing shared by the point-in-time pipelines."""

from datetime import date, datetime
import re

import pandas as pd

FUNDAMENTAL_DATE_COLUMNS = ("period_start", "period_end", "filing_date")


def parse_dates(values: pd.Series, *, allow_missing: bool = False) -> pd.Series:
    """Return naive datetime64[ns]; accept ISO extended/basic dates or date objects.

    Parse each value independently, so valid mixed representations do not depend
    on the first row. Numeric epochs, ambiguous date strings, timezones and
    intraday timestamps are rejected rather than silently normalized.
    """
    parsed = []
    for index, value in values.items():
        try:
            if pd.isna(value):
                if not allow_missing:
                    raise ValueError("Missing date")
                parsed.append(pd.NaT)
                continue
            if isinstance(value, str):
                if not re.fullmatch(r"(?:\d{4}-\d{2}-\d{2}|\d{8})", value):
                    raise ValueError("Expected an ISO calendar date")
                timestamp = pd.Timestamp(date.fromisoformat(value))
            elif isinstance(value, (date, datetime, pd.Timestamp)):
                timestamp = pd.Timestamp(value)
            else:
                raise ValueError("Expected an ISO string or date object")
            if timestamp.tz is not None or timestamp != timestamp.normalize():
                raise ValueError("Expected a timezone-naive date without an intraday time")
            parsed.append(timestamp.as_unit("ns"))
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"Invalid {values.name or 'date'} at row {index}: {value!r}: {exc}") from exc
    return pd.Series(parsed, index=values.index, name=values.name, dtype="datetime64[ns]")


def canonical_dates(values: pd.Series, *, allow_missing: bool = False) -> pd.Series:
    """Stable YYYY-MM-DD representation at CSV/Stage 4 validator boundaries."""
    return parse_dates(values, allow_missing=allow_missing).dt.strftime("%Y-%m-%d")


def normalize_fundamental_dates(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    for column in FUNDAMENTAL_DATE_COLUMNS:
        if column not in result:
            raise ValueError(f"Missing fundamental date column: {column}")
        result[column] = canonical_dates(result[column])
    return result


def date_columns(columns) -> list[str]:
    """Recognize core, reference and state date fields, excluding age/flags."""
    return [c for c in columns if "days_since" not in c and (c == "date" or c in FUNDAMENTAL_DATE_COLUMNS
            or c == "effective_date"
            or c.endswith(("_filing_date", "_period_start", "_period_end", "_effective_date")))]
