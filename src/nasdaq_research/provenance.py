"""Field-aware equality and source-preserving CSV serialization."""

import csv
from io import StringIO
from pathlib import Path

import pandas as pd
import numpy as np

from nasdaq_research.dates import canonical_dates, date_columns, parse_dates

# Derived arithmetic/CSV round trips only. Never apply to raw SEC values,
# identifiers, OHLCV, metadata, information ages, flags or source concept names.
DERIVED_RTOL = 0.0
DERIVED_ATOL = 1e-12


def assert_values_equal(
    actual: pd.DataFrame, expected: pd.DataFrame, *, derived_columns=(),
) -> None:
    """Compare schema/order and values; numeric dtype differences and paired NaNs allowed."""
    if list(actual.columns) != list(expected.columns) or len(actual) != len(expected):
        raise AssertionError("Column order/schema or row count differs")
    actual = actual.copy().reset_index(drop=True)
    expected = expected.copy().reset_index(drop=True)
    for column in date_columns(expected.columns):
        actual[column] = parse_dates(actual[column], allow_missing=True)
        expected[column] = parse_dates(expected[column], allow_missing=True)
    # Normalize nullable scalar sentinels, not values: pd.NA/NaN/None must be
    # equivalent across CSV/object/extension dtypes, with missing positions exact.
    pd.testing.assert_frame_equal(actual.isna(), expected.isna(), check_exact=True)
    for column in expected:
        if expected[column].isna().any() and column not in date_columns(expected.columns):
            actual[column] = actual[column].to_numpy(dtype=object, na_value=np.nan)
            expected[column] = expected[column].to_numpy(dtype=object, na_value=np.nan)
    derived = [c for c in expected if c in set(derived_columns)]
    exact = [c for c in expected if c not in derived]
    pd.testing.assert_frame_equal(actual[exact], expected[exact], check_dtype=False, check_exact=True)
    pd.testing.assert_frame_equal(
        actual[derived], expected[derived], check_dtype=False, check_exact=False,
        rtol=DERIVED_RTOL, atol=DERIVED_ATOL,
    )


def csv_with_source_cells(data: pd.DataFrame, source_path: Path, *, key_columns=()) -> str:
    """Serialize appended columns while preserving exact original source tokens.

    pandas' default decimal parser is not a binary float round trip. Rewriting
    parsed market/SEC decimals can therefore change them on a second read.
    Reuse their original CSV tokens, canonicalizing date tokens only. Source
    values are checked before reuse, so this cannot conceal a mutation.
    """
    source = pd.read_csv(source_path, dtype={c: "string" for c in date_columns(data.columns)})
    with source_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        source_rows = list(reader)
    if list(data.columns[:len(header)]) != header:
        raise ValueError("Output must retain source columns in their original order")
    source = source.reset_index(drop=True)
    for c in date_columns(header):
        source[c] = canonical_dates(source[c], allow_missing=True)
    if key_columns:
        if source.duplicated(list(key_columns)).any():
            raise ValueError("Source CSV contains duplicate keys")
        positions = {key: i for i, key in enumerate(source.loc[:, key_columns].itertuples(index=False, name=None))}
        order = [positions[key] for key in data.loc[:, key_columns].itertuples(index=False, name=None)]
    else:
        if len(data) != len(source):
            raise ValueError("Source CSV row count differs")
        order = list(range(len(source)))
    assert_values_equal(data.loc[:, header], source.iloc[order])
    generated = list(csv.reader(StringIO(data.to_csv(index=False))))
    date_positions = {header.index(c) for c in date_columns(header)}
    result = StringIO(newline="")
    writer = csv.writer(result, lineterminator="\n")
    writer.writerow(generated[0])
    for row, position in zip(generated[1:], order):
        original = source_rows[position]
        writer.writerow([row[i] if i in date_positions else original[i] for i in range(len(header))]
                        + row[len(header):])
    return result.getvalue()
