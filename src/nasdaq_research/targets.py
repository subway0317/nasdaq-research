"""Stage 8: offline next-session-open, execution-aware continuous targets.

X is known after session t closes. Y uses the next observed open and the
observed close at t+h. All X cells and rows are retained, including warm-up NaNs.
"""

import argparse
import csv
import hashlib
from importlib.metadata import version
from io import StringIO
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nasdaq_research.config import DATA_DIR
from nasdaq_research.dates import canonical_dates, date_columns, parse_dates
from nasdaq_research.provenance import (
    DERIVED_ATOL, DERIVED_RTOL, assert_values_equal, csv_with_source_cells,
)

RESEARCH_DIR = DATA_DIR / "research"
HORIZONS = (1, 5, 20)
PRIMARY_TARGET = "forward_return_5d"
RETURN_COLUMNS = tuple(f"forward_return_{h}d" for h in HORIZONS)
TARGET_PAYLOAD_COLUMNS = (
    "target_entry_date", "target_entry_open",
    *(c for h in HORIZONS for c in (f"target_exit_date_{h}d", f"target_exit_close_{h}d", f"forward_return_{h}d")),
)
TARGET_COLUMNS = ("date", "ticker", *TARGET_PAYLOAD_COLUMNS)
TARGET_DATE_COLUMNS = ("date", "target_entry_date", *(f"target_exit_date_{h}d" for h in HORIZONS))
LIMITATIONS = [
    "The current market window is about one year and describes NVDA only.",
    "Adjacent 5-session and 20-session target intervals overlap; rows are not independent observations.",
    "The 20-session target has fewer independent intervals than its daily nonmissing count suggests.",
    "Prices use the existing archived OHLC basis; no new corporate-action or dividend adjustment is made.",
    "Entry/exit prices are hypothetical observed open/close prices; fills and execution are not guaranteed.",
    "Transaction costs, slippage and market impact are not included.",
    "No model, feature selection, feature-target analysis or backtest is performed.",
    "No train/validation/test split is defined. Simple shuffled random splitting is prohibited for later modeling.",
    "Chronological splits, purge and embargo are left for Stage 9; entry/exit provenance is retained for them.",
    "The calendar consists only of observed rows; completeness of exchange-session coverage is inherited from upstream.",
]


def invalid_price_count(features: pd.DataFrame) -> int:
    """Count rows violating the strict numeric, finite, positive open/close contract."""
    invalid = np.zeros(len(features), dtype=bool)
    for column in ("open", "close"):
        if column not in features:
            return len(features)
        values = features[column]
        if (not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)
                or pd.api.types.is_complex_dtype(values)):
            invalid |= np.ones(len(features), dtype=bool)
        else:
            numbers = values.to_numpy(dtype=float, na_value=np.nan)
            invalid |= ~np.isfinite(numbers) | (numbers <= 0)
    return int(invalid.sum())


def _check_features(features: pd.DataFrame) -> pd.Series:
    """Require sorted, unique NVDA sessions; reject invalid input without repair."""
    required = {"date", "ticker", "open", "close"}
    if features.empty or not required <= set(features):
        raise ValueError(f"Nonempty feature matrix requires {sorted(required)}")
    if features.columns.duplicated().any():
        raise ValueError("Duplicate feature column names")
    if not features.ticker.eq("NVDA").fillna(False).all():
        raise ValueError("Stage 8 supports exactly ticker == NVDA; mixed tickers are rejected")
    dates = parse_dates(features.date)
    if dates.duplicated().any():
        raise ValueError("Duplicate ticker/trading-date observations")
    if not dates.is_monotonic_increasing:
        raise ValueError("Trading dates must be ascending; no sorting or row removal is performed")
    if any(c.startswith(("target_", "forward_return_")) for c in features.columns):
        raise ValueError("Source X must not contain target columns")
    bad = invalid_price_count(features)
    if bad:
        raise ValueError(f"Invalid open/close prices in {bad} rows; finite positive numeric prices required")
    return dates


def build_targets(features: pd.DataFrame) -> pd.DataFrame:
    """Construct t+1 open to t+h close labels by position, preserving every row.

    Single-ticker scope is explicit. Missing future sessions leave provenance
    and labels empty; invalid input prices raise rather than being substituted.
    """
    dates = _check_features(features)
    sessions = dates.dt.strftime("%Y-%m-%d")
    entry = features.open.astype(float).shift(-1)
    targets = pd.DataFrame({"date": sessions, "ticker": features.ticker.copy(),
                            "target_entry_date": sessions.shift(-1), "target_entry_open": entry},
                           index=features.index)
    for horizon in HORIZONS:
        exit_price = features.close.astype(float).shift(-horizon)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            returns = exit_price / entry - 1.0
        if np.isinf(returns.to_numpy(dtype=float)).any():
            raise ValueError(f"Non-finite return arithmetic at horizon {horizon}; source prices are not altered")
        targets[f"target_exit_date_{horizon}d"] = sessions.shift(-horizon)
        targets[f"target_exit_close_{horizon}d"] = exit_price
        targets[f"forward_return_{horizon}d"] = returns
    return targets.loc[:, TARGET_COLUMNS]


def _same_scalar(actual: Any, expected: Any) -> bool:
    """Exact price/metadata equality, allowing only paired missing sentinels."""
    if pd.isna(actual) or pd.isna(expected):
        return bool(pd.isna(actual) and pd.isna(expected))
    return bool(actual == expected)


def validate_targets(targets: pd.DataFrame, features: pd.DataFrame,
                     labeled: pd.DataFrame | None = None) -> dict[str, Any]:
    """Audit targets independently with row-by-row source positions and scalar math.

    This validator never calls build_targets. Dates and source prices are exact;
    returns use rtol=0 and the project's derived-arithmetic atol=1e-12.
    """
    checks: dict[str, bool] = {}
    report: dict[str, Any] = {
        "ticker": "NVDA", "row_count": len(targets), "source_feature_row_count": len(features),
        "primary_target": PRIMARY_TARGET, "duplicate_date_count": 0,
        "invalid_price_count": invalid_price_count(features),
        "entry_after_feature_date_violations": 0, "horizon_order_violations": 0,
        "observed_session_position_violations": 0, "price_provenance_violations": 0,
        "return_formula_violations": 0, "formula_tolerance": {"rtol": DERIVED_RTOL, "atol": DERIVED_ATOL},
    }
    errors = []
    try:
        if "date" in features:
            report["duplicate_date_count"] = int(parse_dates(features.date).duplicated().sum())
        dates = _check_features(features).reset_index(drop=True)
        report["date_range"] = {"start": dates.iloc[0].strftime("%Y-%m-%d"),
                                "end": dates.iloc[-1].strftime("%Y-%m-%d")}
        checks["valid_source_contract"] = True
    except (ValueError, TypeError, AttributeError) as exc:
        report.update({"valid": False, "validation_status": "failed", "checks": {"valid_source_contract": False},
                       "errors": [str(exc)]})
        return report
    checks["target_schema_exact"] = list(targets.columns) == list(TARGET_COLUMNS)
    checks["target_row_count_unchanged"] = len(targets) == len(features)
    if not checks["target_schema_exact"] or not checks["target_row_count_unchanged"]:
        report.update({"valid": False, "validation_status": "failed", "checks": checks,
                       "errors": [k for k, passed in checks.items() if not passed]})
        return report
    t = targets.reset_index(drop=True)
    source = features.reset_index(drop=True)
    try:
        target_dates = {c: parse_dates(t[c], allow_missing=c != "date") for c in TARGET_DATE_COLUMNS}
        checks["feature_date_identity"] = target_dates["date"].equals(dates)
    except (ValueError, TypeError) as exc:
        report.update({"valid": False, "validation_status": "failed", "checks": checks, "errors": [str(exc)]})
        return report
    checks["ticker_identity"] = bool(t.ticker.eq(source.ticker).fillna(False).all())
    numeric_columns = ["target_entry_open", *(f"target_exit_close_{h}d" for h in HORIZONS), *RETURN_COLUMNS]
    checks["target_numeric_dtypes"] = all(pd.api.types.is_numeric_dtype(t[c])
                                         and not pd.api.types.is_bool_dtype(t[c])
                                         and not pd.api.types.is_complex_dtype(t[c]) for c in numeric_columns)
    if not checks["target_numeric_dtypes"]:
        report.update({"valid": False, "validation_status": "failed", "checks": checks,
                       "errors": ["Targets and price provenance must be numeric or missing"]})
        return report
    checks["targets_finite_or_missing"] = all(
        not np.isinf(t[c].to_numpy(dtype=float, na_value=np.nan)).any() for c in numeric_columns)
    entry = target_dates["target_entry_date"]
    exits = {h: target_dates[f"target_exit_date_{h}d"] for h in HORIZONS}
    report["entry_after_feature_date_violations"] = int((entry.notna() & entry.le(dates)).sum())
    bad_order = exits[1].notna() & (entry.isna() | exits[1].lt(entry))
    for smaller, larger in zip(HORIZONS, HORIZONS[1:]):
        bad_order |= exits[larger].notna() & (exits[smaller].isna() | exits[larger].le(exits[smaller]))
    report["horizon_order_violations"] = int(bad_order.sum())
    formula_counts = {h: 0 for h in HORIZONS}
    unavailable_counts = {h: 0 for h in HORIZONS}
    # Deliberately scalar indexing; the builder uses vector shifts instead.
    for position in range(len(source)):
        entry_position = position + 1
        expected_entry_date = dates.iloc[entry_position] if entry_position < len(source) else pd.NaT
        expected_entry_open = float(source.open.iloc[entry_position]) if entry_position < len(source) else np.nan
        report["observed_session_position_violations"] += int(not _same_scalar(entry.iloc[position], expected_entry_date))
        report["price_provenance_violations"] += int(not _same_scalar(t.target_entry_open.iloc[position], expected_entry_open))
        for h in HORIZONS:
            exit_position = position + h
            available = exit_position < len(source)
            expected_exit_date = dates.iloc[exit_position] if available else pd.NaT
            expected_close = float(source.close.iloc[exit_position]) if available else np.nan
            report["observed_session_position_violations"] += int(not _same_scalar(exits[h].iloc[position], expected_exit_date))
            close_cell = t[f"target_exit_close_{h}d"].iloc[position]
            report["price_provenance_violations"] += int(not _same_scalar(close_cell, expected_close))
            reported_close = np.nan if pd.isna(close_cell) else float(close_cell)
            return_cell = t[f"forward_return_{h}d"].iloc[position]
            observed_return = np.nan if pd.isna(return_cell) else float(return_cell)
            if not available:
                unavailable_counts[h] += 1
                formula_counts[h] += int(not pd.isna(observed_return))
                continue
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                expected_return = np.divide(expected_close, expected_entry_open) - 1.0
                open_cell = t.target_entry_open.iloc[position]
                reported_open = np.nan if pd.isna(open_cell) else float(open_cell)
                reported_formula = np.divide(reported_close, reported_open) - 1.0
            formula_counts[h] += int(
                not np.isfinite(observed_return)
                or not np.isfinite(expected_return) or not np.isfinite(reported_formula)
                or not np.isclose(observed_return, expected_return, rtol=DERIVED_RTOL, atol=DERIVED_ATOL)
                or not np.isclose(observed_return, reported_formula, rtol=DERIVED_RTOL, atol=DERIVED_ATOL))
    for h in HORIZONS:
        report[f"valid_target_count_{h}d"] = int(t[f"forward_return_{h}d"].notna().sum())
        report[f"missing_target_count_{h}d"] = int(t[f"forward_return_{h}d"].isna().sum())
        report[f"insufficient_future_session_count_{h}d"] = unavailable_counts[h]
        report[f"return_formula_violations_{h}d"] = formula_counts[h]
    report["return_formula_violations"] = sum(formula_counts.values())
    for count in ("entry_after_feature_date_violations", "horizon_order_violations",
                  "observed_session_position_violations", "price_provenance_violations", "return_formula_violations"):
        checks[count.replace("_violations", "_valid")] = report[count] == 0
    checks["tail_missingness_correct"] = all(
        report[f"missing_target_count_{h}d"] == unavailable_counts[h] for h in HORIZONS)
    if labeled is not None:
        checks["labeled_schema_exact"] = list(labeled.columns) == [*features.columns, *TARGET_PAYLOAD_COLUMNS]
        checks["labeled_row_count_unchanged"] = len(labeled) == len(features)
        try:
            assert_values_equal(labeled.loc[:, features.columns], features)
            assert_values_equal(labeled.loc[:, TARGET_PAYLOAD_COLUMNS], targets.loc[:, TARGET_PAYLOAD_COLUMNS],
                                derived_columns=RETURN_COLUMNS)
            checks["all_feature_cells_preserved"] = True
            checks["labeled_targets_match"] = True
        except (AssertionError, KeyError, ValueError, TypeError) as exc:
            checks["all_feature_cells_and_targets_preserved"] = False
            errors.append(str(exc))
    errors.extend(k for k, passed in checks.items() if not passed)
    report.update({"valid": not errors, "validation_status": "passed" if not errors else "failed",
                   "checks": checks, "errors": errors})
    return report


def build_labeled_dataset(features: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Append only the eleven Y/provenance columns, preserving the entire X frame."""
    audit = validate_targets(targets, features)
    if not audit["valid"]:
        raise ValueError(f"Invalid target alignment: {audit['errors']}")
    payload = pd.DataFrame({c: targets[c].array for c in TARGET_PAYLOAD_COLUMNS}, index=features.index)
    return pd.concat([features.copy(deep=True), payload], axis=1)


def target_summary(targets: pd.DataFrame) -> pd.DataFrame:
    """Describe Y alone; no feature comparisons, transforms or outlier processing."""
    rows = []
    for h in HORIZONS:
        name = f"forward_return_{h}d"
        values = targets[name]
        quantiles = values.quantile([.01, .05, .25, .5, .75, .95, .99])
        rows.append({"target_name": name, "horizon_observed_sessions": h, "primary_target": name == PRIMARY_TARGET,
                     "count": int(values.count()), "missing_count": int(values.isna().sum()),
                     "mean": values.mean(), "std": values.std(ddof=1), "min": values.min(),
                     "p01": quantiles.loc[.01], "p05": quantiles.loc[.05], "p25": quantiles.loc[.25],
                     "median": quantiles.loc[.5], "p75": quantiles.loc[.75], "p95": quantiles.loc[.95],
                     "p99": quantiles.loc[.99], "max": values.max()})
    return pd.DataFrame(rows)


def _read_csv(path: Path | StringIO, *, artifact: bool = False) -> pd.DataFrame:
    """Match upstream source parsing; round-trip newly derived return decimals."""
    columns = pd.read_csv(path, nrows=0).columns
    if isinstance(path, StringIO):
        path.seek(0)
    strings = set(date_columns(columns)) | set(TARGET_DATE_COLUMNS) | {"ticker"}
    strings.update(c for c in columns if c == "accession" or c.endswith("_accession"))
    converters = {c: lambda value: float(value) if value else np.nan for c in RETURN_COLUMNS if artifact and c in columns}
    return pd.read_csv(path, dtype={c: "string" for c in strings if c in columns}, converters=converters)


def _csv_with_target_prices(frame: pd.DataFrame, source_path: Path, *, base_csv: str | None = None) -> str:
    """Copy exact future source price tokens into Y provenance to avoid parser drift."""
    with source_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        source_header, source_rows = next(reader), list(reader)
    rows = list(csv.reader(StringIO(base_csv if base_csv is not None else frame.to_csv(index=False))))
    header = rows[0]
    if len(rows) - 1 != len(source_rows):
        raise ValueError("Source and output row counts differ during serialization")
    bindings = {"target_entry_open": (1, source_header.index("open")),
                **{f"target_exit_close_{h}d": (h, source_header.index("close")) for h in HORIZONS}}
    output = StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    for position, row in enumerate(rows[1:]):
        for name, (offset, source_column) in bindings.items():
            row[header.index(name)] = source_rows[position + offset][source_column] if position + offset < len(source_rows) else ""
        writer.writerow(row)
    return output.getvalue()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_upstream(features: pd.DataFrame, features_path: Path,
                    feature_validation_path: Path, diagnostics_path: Path) -> None:
    """Require successful upstream reports tied to this exact feature artifact."""
    stage6 = json.loads(feature_validation_path.read_text(encoding="utf-8"))
    stage7 = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    if stage6.get("validation", {}).get("valid") is not True or stage7.get("validation", {}).get("valid") is not True:
        raise ValueError("Stage 6.1 and Stage 7 validation reports must pass")
    dates = canonical_dates(features.date)
    expected = {"ticker": "NVDA", "row_count": len(features), "column_count": len(features.columns),
                "date_range": {"start": dates.iloc[0], "end": dates.iloc[-1]}}
    if any(stage7.get(k) != value for k, value in expected.items()):
        raise ValueError("Stage 7 report does not describe the input feature matrix")
    if stage7.get("source_sha256_after", {}).get(str(features_path)) != _sha256(features_path):
        raise ValueError("Source feature hash differs from the Stage 7 checkpoint")
    if stage6.get("ticker") != "NVDA" or stage6.get("final_feature_rows") != len(features) or stage6.get("final_matrix_columns") != len(features.columns):
        raise ValueError("Stage 6.1 report shape/ticker differs from input")


def run_pipeline(
    features_path: Path = RESEARCH_DIR / "NVDA_features.csv",
    output_dir: Path = RESEARCH_DIR / "targets",
    labeled_path: Path = RESEARCH_DIR / "NVDA_labeled.csv",
    feature_validation_path: Path = RESEARCH_DIR / "NVDA_feature_validation.json",
    diagnostics_path: Path = RESEARCH_DIR / "diagnostics" / "NVDA_feature_diagnostics.json",
) -> dict[str, Any]:
    """Read existing X, validate Y in memory and after CSV reload, write new artifacts."""
    features_path, output_dir, labeled_path, feature_validation_path, diagnostics_path = (
        Path(p).resolve() for p in (features_path, output_dir, labeled_path, feature_validation_path, diagnostics_path))
    inputs = (features_path, feature_validation_path, diagnostics_path)
    if any(output_dir == p or output_dir in p.parents for p in inputs) or labeled_path in inputs:
        raise ValueError("Stage 8 outputs must not overwrite or contain source artifacts")
    if labeled_path.name != "NVDA_labeled.csv":
        raise ValueError("Labeled artifact must be named NVDA_labeled.csv to protect earlier artifacts")
    target_path = output_dir / "NVDA_targets.csv"
    summary_path = output_dir / "target_summary.csv"
    validation_path = output_dir / "NVDA_target_validation.json"
    figure_paths = [output_dir / "figures" / f"{name}_distribution.png" for name in RETURN_COLUMNS]
    figure_paths.append(output_dir / "figures" / "forward_return_5d_time_series.png")
    outputs = {target_path, summary_path, validation_path, labeled_path, *figure_paths}
    if len(outputs) != 4 + len(figure_paths) or outputs & set(inputs):
        raise ValueError("Output paths must be distinct and separate from inputs")
    if any(p.is_symlink() for p in outputs) or (output_dir / "figures").is_symlink():
        raise ValueError("Stage 8 output symlinks are rejected to protect source artifacts")
    protected = set(inputs)
    for root in {DATA_DIR.resolve(), features_path.parent}:
        if root.exists():
            protected.update(p.resolve() for p in root.rglob("*") if p.is_file() and p.resolve() not in outputs)
    before = {str(p): _sha256(p) for p in sorted(protected)}
    features = _read_csv(features_path)
    original = features.copy(deep=True)
    _check_features(features)
    _check_upstream(features, features_path, feature_validation_path, diagnostics_path)
    targets = build_targets(features)
    labeled = build_labeled_dataset(features, targets)
    pd.testing.assert_frame_equal(features, original, check_exact=True)
    target_csv = _csv_with_target_prices(targets, features_path)
    labeled_csv = _csv_with_target_prices(labeled, features_path,
                                          base_csv=csv_with_source_cells(labeled, features_path))
    # Audit the exact bytes to be written before publishing any CSV artifacts.
    serialized_audit = validate_targets(_read_csv(StringIO(target_csv), artifact=True), features,
                                       _read_csv(StringIO(labeled_csv), artifact=True))
    if not serialized_audit["valid"]:
        raise ValueError(f"CSV round-trip validation failed: {serialized_audit['errors']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    labeled_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(target_csv, encoding="utf-8")
    labeled_path.write_text(labeled_csv, encoding="utf-8")
    target_summary(targets).to_csv(summary_path, index=False)
    from nasdaq_research.target_figures import generate_target_figures
    generated_figures = generate_target_figures(targets, output_dir / "figures")
    report = validate_targets(_read_csv(target_path, artifact=True), features,
                              _read_csv(labeled_path, artifact=True))
    after = {path: _sha256(Path(path)) for path in before}
    report["checks"].update({"source_dataframe_unchanged": features.equals(original),
                             "source_artifact_hashes_unchanged": before == after,
                             "csv_roundtrip_validated_before_write": serialized_audit["valid"],
                             "expected_figures_generated": generated_figures == figure_paths})
    report.update({
        "source_feature_file": str(features_path), "source_feature_sha256": before[str(features_path)],
        "source_feature_column_count": len(features.columns), "labeled_column_count": len(labeled.columns),
        "source_artifact_sha256_before": before, "source_artifact_sha256_after": after,
        "source_artifacts_unchanged": before == after,
        "feature_timestamp": "After the close of observed trading session t",
        "entry_rule": "Open of the first observed trading session strictly after feature date (t+1)",
        "horizon_basis": "Position in the existing sorted observed-session rows; no calendar-day offsets",
        "target_definitions": {f"forward_return_{h}d": f"Close[t+{h}] / Open[t+1] - 1" for h in HORIZONS},
        "price_policy": "Require all input open/close prices numeric, finite and strictly positive; reject without repair",
        "missing_target_policy": "Insufficient future sessions remain NaN; no feature row is removed",
        "feature_target_boundary": "Future prices are confined to the eleven appended Y/provenance columns; X is preserved",
        "overlapping_targets": {"horizons": [h for h in HORIZONS if h > 1],
                                "interval_columns": ["target_entry_date", *(f"target_exit_date_{h}d" for h in HORIZONS)],
                                "purge_embargo_implemented": False},
        "limitations": LIMITATIONS,
        "output_paths": {"targets": str(target_path), "labeled": str(labeled_path),
                         "summary": str(summary_path), "validation": str(validation_path),
                         "figures": [str(p) for p in generated_figures]},
        "reproducibility": {"randomness": "none", "generated_timestamp": "omitted",
                            "source_tokens": "Original X and future source price decimals preserved",
                            "return_parser": "Python float decimal round trip for derived target returns",
                            "pandas_version": pd.__version__, "numpy_version": np.__version__,
                            "matplotlib_version": version("matplotlib")},
        "output_artifact_sha256": {str(p): _sha256(p) for p in [target_path, labeled_path, summary_path, *generated_figures]},
    })
    report["errors"].extend(k for k, passed in report["checks"].items() if not passed and k not in report["errors"])
    report["valid"] = not report["errors"]
    report["validation_status"] = "passed" if report["valid"] else "failed"
    validation_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not report["valid"]:
        raise ValueError(f"Target validation failed: {report['errors']}")
    return report


def terminal_summary(report: dict[str, Any]) -> str:
    """Concise CLI summary of actual source shape, label counts and validation."""
    lines = [f"Ticker: {report['ticker']}", f"Rows: {report['row_count']}",
             f"Source columns / labeled columns: {report['source_feature_column_count']} / {report['labeled_column_count']}",
             f"Date range: {report['date_range']['start']} to {report['date_range']['end']}",
             f"Primary target: {report['primary_target']}", "Entry: next observed trading session open"]
    lines.extend(f"forward_return_{h}d valid / missing: {report[f'valid_target_count_{h}d']} / {report[f'missing_target_count_{h}d']}"
                 for h in HORIZONS)
    lines.extend([f"Entry / horizon / formula violations: {report['entry_after_feature_date_violations']} / "
                  f"{report['horizon_order_violations']} / {report['return_formula_violations']}",
                  f"Duplicate dates / invalid prices: {report['duplicate_date_count']} / {report['invalid_price_count']}",
                  f"Source artifacts unchanged: {report['source_artifacts_unchanged']}",
                  f"Targets: {report['output_paths']['targets']}", f"Labeled: {report['output_paths']['labeled']}",
                  f"Validation: {report['validation_status']}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=RESEARCH_DIR / "NVDA_features.csv")
    parser.add_argument("--output-dir", type=Path, default=RESEARCH_DIR / "targets")
    parser.add_argument("--labeled-path", type=Path, default=RESEARCH_DIR / "NVDA_labeled.csv")
    parser.add_argument("--feature-validation-path", type=Path, default=RESEARCH_DIR / "NVDA_feature_validation.json")
    parser.add_argument("--diagnostics-path", type=Path, default=RESEARCH_DIR / "diagnostics" / "NVDA_feature_diagnostics.json")
    args = parser.parse_args()
    report = run_pipeline(args.features_path, args.output_dir, args.labeled_path, args.feature_validation_path, args.diagnostics_path)
    print(terminal_summary(report))


if __name__ == "__main__":
    main()
