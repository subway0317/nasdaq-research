"""Stage 6: filing-level ratios and comparable-period YoY, mapped offline."""

import argparse
import hashlib
import json
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nasdaq_research.alignment import (
    AGE_COLUMNS, RESEARCH_DATA_DIR, build_research_dataset, effective_dates,
    validate_research_dataset,
)
from nasdaq_research.config import DATA_DIR, FUNDAMENTALS_DATA_DIR
from nasdaq_research.features import PROCESSED_COLUMNS, WARMUP_ROWS, read_processed_history, validate_features
from nasdaq_research.fundamentals import KEY_COLUMNS, METADATA_COLUMNS, validate_fundamentals
from nasdaq_research.dates import (FUNDAMENTAL_DATE_COLUMNS, canonical_dates,
                                  date_columns, normalize_fundamental_dates, parse_dates)
from nasdaq_research.provenance import assert_values_equal, csv_with_source_cells, DERIVED_ATOL, DERIVED_RTOL
from nasdaq_research.comparative import comparative_diagnostic

RATIO_FIELDS = {
    "gross_margin": "gross_profit",
    "operating_margin": "operating_income",
    "net_margin": "net_income",
    "rd_to_revenue": "research_and_development",
    "sga_to_revenue": "selling_general_administrative",
    "ocf_margin": "operating_cash_flow",
    "fcf_margin": "free_cash_flow",
}
BALANCE_RATIOS = {
    "current_ratio": ("current_assets", "current_liabilities"),
    "cash_to_assets": ("cash_and_cash_equivalents", "total_assets"),
    "liabilities_to_assets": ("total_liabilities", "total_assets"),
    "debt_to_assets": ("long_term_debt", "total_assets"),
}
GROWTH_FIELDS = {
    "revenue": "revenue",
    "gross_profit": "gross_profit",
    "operating_income": "operating_income",
    "net_income": "net_income",
    "rd": "research_and_development",
    "operating_cash_flow": "operating_cash_flow",
}
QUARTERLY_RATIO_FEATURES = tuple(f"q_{name}" for name in RATIO_FIELDS)
ANNUAL_RATIO_FEATURES = tuple(f"fy_{name}" for name in RATIO_FIELDS)
QUARTERLY_GROWTH_FEATURES = tuple(f"q_{name}_growth_yoy" for name in GROWTH_FIELDS)
ANNUAL_GROWTH_FEATURES = tuple(f"fy_{name}_growth_yoy" for name in GROWTH_FIELDS)
FEATURE_COLUMNS = (
    *QUARTERLY_RATIO_FEATURES, *ANNUAL_RATIO_FEATURES, *BALANCE_RATIOS,
    *QUARTERLY_GROWTH_FEATURES, *ANNUAL_GROWTH_FEATURES,
)
REFERENCE_FIELDS = tuple(c for c in METADATA_COLUMNS if c != "ticker")
REFERENCE_COLUMNS = tuple(f"yoy_reference_{c}" for c in REFERENCE_FIELDS)
PROVENANCE_COLUMNS = ("yoy_comparison_status", *REFERENCE_COLUMNS)
SNAPSHOT_PAYLOAD_COLUMNS = (*FEATURE_COLUMNS, *PROVENANCE_COLUMNS)
BALANCE_FIELDS = ("cash_and_cash_equivalents", "total_assets", "total_liabilities",
                  "stockholders_equity", "current_assets", "current_liabilities", "long_term_debt")
STATE_FIELDS = (*REFERENCE_FIELDS, "effective_date", "effective_date_is_sample_truncated",
                "effective_date_basis", *AGE_COLUMNS)
STATE_METADATA_COLUMNS = tuple(f"{prefix}_{c}" for prefix in ("q", "fy", "bs") for c in STATE_FIELDS)
STATE_REFERENCE_COLUMNS = tuple(f"{prefix}_{c}" for prefix in ("q", "fy") for c in PROVENANCE_COLUMNS)
GLOBAL_EFFECTIVE_COLUMNS = ("fundamental_effective_date_is_sample_truncated", "fundamental_effective_date_basis")
PAYLOAD_COLUMNS = (*SNAPSHOT_PAYLOAD_COLUMNS, *STATE_METADATA_COLUMNS,
                   *STATE_REFERENCE_COLUMNS, *(f"bs_{c}" for c in BALANCE_FIELDS), *GLOBAL_EFFECTIVE_COLUMNS)
DEFAULT_FUNDAMENTALS = FUNDAMENTALS_DATA_DIR / "processed" / "NVDA_fundamentals.csv"
DEFAULT_RESEARCH = RESEARCH_DATA_DIR / "NVDA_research.csv"
DEFAULT_RAW = FUNDAMENTALS_DATA_DIR / "raw" / "NVDA_companyfacts.json"


def _divide(numerator: float, denominator: float, *, growth: bool = False) -> float:
    """Require a positive denominator; missing/overflow stays missing."""
    if pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
        return np.nan
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        value = np.divide(float(numerator), float(denominator)) - int(growth)
    return float(value) if np.isfinite(value) else np.nan


def _reference(row: pd.Series, known: pd.DataFrame) -> tuple[pd.Series | None, str]:
    """Resolve a whole prior-year observation using only this filing's vintage."""
    same_basis = known[known.data_period_type.eq(row.data_period_type)
                       & known.fiscal_period.eq(row.fiscal_period)]
    current_key = same_basis[same_basis.fiscal_year.eq(row.fiscal_year)]
    if len(current_key.drop_duplicates(["period_start", "period_end"])) > 1:
        return None, "ambiguous_current_period"
    previous = same_basis[same_basis.fiscal_year.eq(int(row.fiscal_year) - 1)]
    if previous.empty:
        return None, "missing_previous_period"
    if len(previous.drop_duplicates(["period_start", "period_end"])) > 1:
        return None, "ambiguous_previous_period"
    previous = previous.assign(_amended=previous.form.str.endswith("/A").astype(int))
    reference = previous.sort_values([
        "filing_date", "period_end", "_amended", "period_start", "accession",
    ], kind="stable").iloc[-1]
    # Fiscal labels alone can be wrong in archived Company Facts. Allow 52/53-week
    # years, but reject adjacent quarters or skipped years carrying matching labels.
    gaps = [(pd.Timestamp(row[c]) - pd.Timestamp(reference[c])).days
            for c in ("period_start", "period_end")]
    if not all(330 <= gap <= 400 for gap in gaps) or reference.period_end >= row.period_start:
        return None, "noncomparable_period_dates"
    return reference, "available"


def build_fundamental_features(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Compute frozen snapshots at each filing; no daily pct_change or YTD flows."""
    fundamentals = normalize_fundamental_dates(fundamentals)
    quality = validate_fundamentals(fundamentals)
    if not quality["valid"]:
        raise ValueError(f"Invalid fundamentals: {quality['errors']}")
    reserved = set(PAYLOAD_COLUMNS) | {"effective_date", "selected_for_daily", "selected_for_stage5",
                "selected_for_q", "selected_for_fy", "selected_for_bs",
                "effective_date_basis", "effective_date_is_sample_truncated"}
    if reserved & set(fundamentals):
        raise ValueError("Fundamentals already contain Stage 6 reserved columns")
    eligible = fundamentals[fundamentals.data_period_type.isin(["quarterly", "annual"])].copy()
    # All sorting, cutoff and period comparisons operate on typed dates. The
    # canonical strings above are only the boundary of the Stage 4 validator.
    for column in FUNDAMENTAL_DATE_COLUMNS:
        eligible[column] = parse_dates(eligible[column])
    eligible = eligible.sort_values([
        "filing_date", "period_end", "period_start", "accession", "data_period_type",
    ], kind="stable").reset_index(drop=True)
    computed = []
    for _, row in eligible.iterrows():
        values: dict[str, Any] = {c: np.nan for c in FEATURE_COLUMNS}
        prefix = "q" if row.data_period_type == "quarterly" else "fy"
        for name, numerator in RATIO_FIELDS.items():
            values[f"{prefix}_{name}"] = _divide(row[numerator], row.revenue)
        for name, (numerator, denominator) in BALANCE_RATIOS.items():
            values[name] = _divide(row[numerator], row[denominator])
        known = eligible[eligible.filing_date.le(row.filing_date)]
        reference, status = _reference(row, known)
        values["yoy_comparison_status"] = status
        values.update({c: None for c in REFERENCE_COLUMNS})
        if reference is not None:
            if reference.filing_date > row.filing_date:
                raise ValueError("Historical reference filing is later than current filing")
            values.update({f"yoy_reference_{c}": reference[c] for c in REFERENCE_FIELDS})
            for name, field in GROWTH_FIELDS.items():
                values[f"{prefix}_{name}_growth_yoy"] = _divide(row[field], reference[field], growth=True)
        computed.append(values)
    derived = pd.DataFrame(computed, columns=SNAPSHOT_PAYLOAD_COLUMNS)
    derived = derived.astype({c: float for c in FEATURE_COLUMNS})
    result = pd.concat([eligible, derived], axis=1)
    for column in date_columns(result.columns):
        result[column] = canonical_dates(result[column], allow_missing=True)
    return result


def _canonical_research(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize CSV date/nullable-age representation without altering values."""
    result = data.copy().reset_index(drop=True)
    result["date"] = parse_dates(result.date).dt.date
    for column in date_columns(result.columns):
        if column != "date":
            result[column] = (parse_dates(result[column], allow_missing=True) if column.endswith("effective_date")
                              else canonical_dates(result[column], allow_missing=True))
    for column in AGE_COLUMNS:
        result[column] = result[column].astype("Int64")
    return result


def _assert_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    assert_values_equal(actual, expected, derived_columns=FEATURE_COLUMNS)


def _market_from_research(research: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    missing = set(PROCESSED_COLUMNS) - set(research)
    if missing:
        raise ValueError(f"Missing Stage 3 columns: {sorted(missing)}")
    if set(PAYLOAD_COLUMNS) & set(research):
        raise ValueError("Research input already contains Stage 6 columns")
    if not validate_features(research.assign(date=parse_dates(research.date).dt.date))["valid"]:
        raise ValueError("Stage 3 market feature validation failed")
    fundamental_columns = {
        f"fundamental_{c}" if c in METADATA_COLUMNS else c
        for c in fundamentals if c != "ticker"
    } | {"fundamental_effective_date", *AGE_COLUMNS}
    market = research.drop(columns=list(fundamental_columns & set(research))).copy()
    quality = validate_research_dataset(_canonical_research(research), market, fundamentals)
    if not quality["valid"]:
        raise ValueError(f"Stage 5 source alignment failed: {quality['errors']}")
    return market


def build_feature_matrix(
    research: pd.DataFrame, fundamentals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return all eligible historical snapshots and the augmented Stage 5 matrix."""
    fundamentals = normalize_fundamental_dates(fundamentals)
    snapshots = build_fundamental_features(fundamentals)
    market = _market_from_research(research, fundamentals)
    # Keep YTD rows in the enriched input solely for the existing Stage 5 selector
    # to exclude them. This also supports all-YTD/no-effective-filing inputs.
    enriched = fundamentals.merge(
        snapshots.loc[:, [*KEY_COLUMNS, *SNAPSHOT_PAYLOAD_COLUMNS]],
        on=list(KEY_COLUMNS), how="left", validate="one_to_one",
    )
    aligned = build_research_dataset(market, enriched)
    daily = research.copy().reset_index(drop=True)
    payload = {c: aligned[c] for c in PROVENANCE_COLUMNS}
    snapshots["effective_date"] = effective_dates(snapshots.filing_date, parse_dates(research.date))
    truncated, basis = _effective_semantics(snapshots.filing_date, snapshots.effective_date, research.date)
    snapshots["effective_date_is_sample_truncated"] = truncated
    snapshots["effective_date_basis"] = basis
    snapshots["selected_for_stage5"] = _selected(snapshots, aligned)
    masks = {
        "q": enriched.data_period_type.eq("quarterly"),
        "fy": enriched.data_period_type.eq("annual"),
        # Available means at least one BS source field is present. Select this
        # complete observation even when some ratios/metrics are missing.
        "bs": enriched.data_period_type.isin(["quarterly", "annual"])
              & enriched.loc[:, BALANCE_FIELDS].notna().any(axis=1),
    }
    groups = {"q": (*QUARTERLY_RATIO_FEATURES, *QUARTERLY_GROWTH_FEATURES),
              "fy": (*ANNUAL_RATIO_FEATURES, *ANNUAL_GROWTH_FEATURES), "bs": tuple(BALANCE_RATIOS)}
    for prefix, mask in masks.items():
        state = build_research_dataset(market, enriched.loc[mask], allow_empty=True)
        for c in groups[prefix]:
            payload[c] = state[c]
        for c in REFERENCE_FIELDS:
            payload[f"{prefix}_{c}"] = state[f"fundamental_{c}"]
        payload[f"{prefix}_effective_date"] = state.fundamental_effective_date
        flags, labels = _effective_semantics(state.fundamental_filing_date, state.fundamental_effective_date, research.date)
        payload[f"{prefix}_effective_date_is_sample_truncated"] = flags
        payload[f"{prefix}_effective_date_basis"] = labels
        for c in AGE_COLUMNS:
            payload[f"{prefix}_{c}"] = state[c]
        if prefix in ("q", "fy"):
            for c in PROVENANCE_COLUMNS:
                payload[f"{prefix}_{c}"] = state[c]
        else:
            for c in BALANCE_FIELDS:
                payload[f"bs_{c}"] = state[c]
        snapshots[f"selected_for_{prefix}"] = _selected(snapshots, state)
    payload[GLOBAL_EFFECTIVE_COLUMNS[0]], payload[GLOBAL_EFFECTIVE_COLUMNS[1]] = _effective_semantics(
        aligned.fundamental_filing_date, aligned.fundamental_effective_date, research.date)
    daily = pd.concat([daily, pd.DataFrame(payload).loc[:, PAYLOAD_COLUMNS]], axis=1)
    snapshots["selected_for_daily"] = snapshots[["selected_for_q", "selected_for_fy", "selected_for_bs"]].any(axis=1)
    return snapshots, daily


def _selected(snapshots: pd.DataFrame, aligned: pd.DataFrame) -> list[bool]:
    selected = set(zip(*(aligned[f"fundamental_{c}"] for c in KEY_COLUMNS)))
    return [key in selected for key in snapshots.loc[:, KEY_COLUMNS].itertuples(index=False, name=None)]


def _effective_semantics(filed, effective, market_dates):
    filed = parse_dates(filed, allow_missing=True)
    effective = parse_dates(effective, allow_missing=True)
    flags = (filed < parse_dates(market_dates).iloc[0]).astype("boolean")
    flags[filed.isna()] = pd.NA
    labels = pd.Series("observed_next_session", index=filed.index, dtype=object)
    labels[flags.fillna(False)] = "sample_start_truncated"
    labels[effective.isna()] = "no_observed_next_session"
    labels[filed.isna()] = None
    return flags, labels


def _numeric_checks(data: pd.DataFrame) -> dict[str, bool]:
    present = set(FEATURE_COLUMNS).issubset(data)
    numeric = present and all(pd.api.types.is_numeric_dtype(data[c])
                              and not pd.api.types.is_bool_dtype(data[c]) for c in FEATURE_COLUMNS)
    numbers = data.select_dtypes(include="number")
    return {
        "all_feature_columns_present": present,
        "features_numeric_or_missing": numeric,
        "no_infinity": not np.isinf(numbers.to_numpy(dtype=float, na_value=np.nan)).any(),
    }


def validate_feature_matrix(
    daily: pd.DataFrame, snapshots: pd.DataFrame,
    research: pd.DataFrame, fundamentals: pd.DataFrame,
) -> dict[str, Any]:
    """Audit Stage 5 preservation and full filing-level feature/provenance lineage."""
    checks = {f"daily_{k}": bool(v) for k, v in _numeric_checks(daily).items()}
    checks.update({f"snapshot_{k}": bool(v) for k, v in _numeric_checks(snapshots).items()})
    checks["market_row_count_unchanged"] = len(daily) == len(research)
    errors = []
    try:
        assert_values_equal(daily.loc[:, research.columns], research)
        checks["stage5_columns_preserved"] = True
        expected_snapshots, expected_daily = build_feature_matrix(research, fundamentals)
        _assert_equal(snapshots, expected_snapshots)
        checks["snapshots_match_filing_sources"] = True
        _assert_equal(daily, expected_daily)
        checks["daily_matches_point_in_time_snapshots"] = True
    except (ValueError, KeyError, TypeError, AssertionError) as exc:
        errors.append(f"Source/period/alignment validation failed: {exc}")
    try:
        dates = parse_dates(daily.date)
        filed = parse_dates(daily.fundamental_filing_date, allow_missing=True)
        effective = parse_dates(daily.fundamental_effective_date, allow_missing=True)
        checks["filing_strictly_before_trading_date"] = not (filed >= dates).any()
        checks["effective_date_no_later_than_trading_date"] = not (effective > dates).any()
        for prefix, expected_basis in (("q", "quarterly"), ("fy", "annual"), ("bs", None)):
            filed = parse_dates(daily[f"{prefix}_filing_date"], allow_missing=True)
            effective = parse_dates(daily[f"{prefix}_effective_date"], allow_missing=True)
            checks[f"{prefix}_filing_strictly_before_trading_date"] = not (filed >= dates).any()
            checks[f"{prefix}_effective_date_no_later_than_trading_date"] = not (effective > dates).any()
            if expected_basis:
                checks[f"{prefix}_period_basis_separate"] = daily.loc[daily[f"{prefix}_accession"].notna(), f"{prefix}_data_period_type"].eq(expected_basis).all()
                checks[f"{prefix}_reference_known_at_state_filing"] = not (
                    parse_dates(daily[f"{prefix}_yoy_reference_filing_date"], allow_missing=True) > filed).any()
        q_values = snapshots.loc[:, [*QUARTERLY_RATIO_FEATURES, *QUARTERLY_GROWTH_FEATURES]].notna().any(axis=1)
        fy_values = snapshots.loc[:, [*ANNUAL_RATIO_FEATURES, *ANNUAL_GROWTH_FEATURES]].notna().any(axis=1)
        checks["snapshot_period_basis_separate"] = (
            snapshots.loc[q_values, "data_period_type"].eq("quarterly").all()
            and snapshots.loc[fy_values, "data_period_type"].eq("annual").all()
        )
        references = snapshots[snapshots.yoy_reference_accession.notna()]
        checks["reference_period_matches"] = (
            references.yoy_reference_fiscal_year.eq(references.fiscal_year - 1).all()
            and references.yoy_reference_fiscal_period.eq(references.fiscal_period).all()
            and references.yoy_reference_data_period_type.eq(references.data_period_type).all()
        )
        checks["reference_filing_known_at_snapshot"] = (
            parse_dates(references.yoy_reference_filing_date)
            <= parse_dates(references.filing_date)
        ).all()
    except (ValueError, KeyError, AttributeError, TypeError) as exc:
        errors.append(f"Timing/period validation failed: {exc}")
    errors.extend(name for name, passed in checks.items() if not passed)
    return {"valid": not errors, "errors": errors, "checks": {k: bool(v) for k, v in checks.items()}}


def run_pipeline(
    fundamentals_path: Path = DEFAULT_FUNDAMENTALS,
    research_path: Path = DEFAULT_RESEARCH,
    output_dir: Path = RESEARCH_DATA_DIR,
    raw_path: Path | None = DEFAULT_RAW,
) -> dict[str, Any]:
    """Read local Stage 4/5 artifacts, validate and save Stage 6 outputs only."""
    paths = {
        "fundamental_feature_snapshot_path": output_dir / "NVDA_fundamental_features.csv",
        "final_feature_matrix_path": output_dir / "NVDA_features.csv",
        "validation_path": output_dir / "NVDA_feature_validation.json",
    }
    stage1_5_paths = [*sorted((DATA_DIR / "raw").glob("*.csv")),
                      *sorted((DATA_DIR / "processed").glob("*.csv")),
                      *sorted((FUNDAMENTALS_DATA_DIR / "raw").glob("*.json")),
                      *sorted((FUNDAMENTALS_DATA_DIR / "processed").glob("*")),
                      DEFAULT_RESEARCH, RESEARCH_DATA_DIR / "NVDA_validation.json"]
    protected = {p.resolve() for p in stage1_5_paths} | {fundamentals_path.resolve(), research_path.resolve()}
    artifact_hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in stage1_5_paths if p.is_file()}
    if raw_path is not None:
        protected.add(raw_path.resolve())
    if any(p.resolve() in protected for p in paths.values()):
        raise ValueError("Stage 6 outputs must not overwrite inputs")
    input_paths = [fundamentals_path, research_path]
    if raw_path is not None and raw_path.is_file():
        input_paths.append(raw_path)
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
    fundamentals = pd.read_csv(fundamentals_path, dtype={c: "string" for c in FUNDAMENTAL_DATE_COLUMNS})
    normalization = {c: int(fundamentals[c].ne(canonical_dates(fundamentals[c])).sum()) for c in FUNDAMENTAL_DATE_COLUMNS}
    fundamentals = normalize_fundamental_dates(fundamentals)
    research = _canonical_research(pd.read_csv(research_path, dtype={
        c: "string" for c in ("date", "fundamental_filing_date", "fundamental_period_start",
                              "fundamental_period_end", "fundamental_effective_date")}))
    snapshots, daily = build_feature_matrix(research, fundamentals)
    validation = validate_feature_matrix(daily, snapshots, research, fundamentals)
    if not validation["valid"]:
        raise ValueError(f"Stage 6 validation failed: {validation['errors']}")
    snapshot_csv = csv_with_source_cells(snapshots, fundamentals_path, key_columns=KEY_COLUMNS)
    daily_csv = csv_with_source_cells(daily, research_path)
    roundtrip = validate_feature_matrix(
        pd.read_csv(StringIO(daily_csv), dtype={c: float for c in FEATURE_COLUMNS}),
        pd.read_csv(StringIO(snapshot_csv), dtype={c: float for c in FEATURE_COLUMNS}),
        research, fundamentals,
    )
    if not roundtrip["valid"]:
        raise ValueError(f"Stage 6 CSV round-trip validation failed: {roundtrip['errors']}")
    validation["checks"]["csv_roundtrip_preserves_source_values"] = True
    # Verify the existing Stage 5 checkpoint without writing it or rerunning any
    # earlier ingestion/feature stages. Exact bytes include market serialization.
    stage5_reproduction = {"status": "source_alignment_validated", "csv_byte_identical": None}
    if fundamentals_path.resolve() == DEFAULT_FUNDAMENTALS.resolve() and research_path.resolve() == DEFAULT_RESEARCH.resolve():
        rebuilt = build_research_dataset(read_processed_history("NVDA"), fundamentals)
        identical = rebuilt.to_csv(index=False).encode("utf-8") == research_path.read_bytes()
        if not identical:
            raise ValueError("Stage 5 in-memory reproduction differs from existing checkpoint")
        stage5_reproduction = {"status": "reproduced_in_memory", "csv_byte_identical": True}
    issues = snapshots[snapshots.yoy_comparison_status.isin([
        "ambiguous_current_period", "ambiguous_previous_period", "noncomparable_period_dates",
    ])]
    diagnostic = comparative_diagnostic(snapshots, fundamentals, raw_path)
    streams = {}
    for prefix in ("q", "fy", "bs"):
        metadata = [f"{prefix}_{c}" for c in ("accession", "filing_date", "effective_date", "period_start", "period_end", "data_period_type")]
        changes = daily.loc[daily[f"{prefix}_accession"].notna(), metadata].drop_duplicates()
        period_ends = parse_dates(changes[f"{prefix}_period_end"])
        streams[prefix] = {
            "selected_snapshot_count": int(snapshots[f"selected_for_{prefix}"].sum()),
            "selected_sources": changes.assign(**{f"{prefix}_effective_date": canonical_dates(changes[f"{prefix}_effective_date"])}).to_dict("records"),
            "rows_with_state": int(daily[f"{prefix}_accession"].notna().sum()),
            "sample_truncated_rows": int(daily[f"{prefix}_effective_date_is_sample_truncated"].fillna(False).sum()),
            "period_end_backward_transitions": int((period_ends.diff().dt.days < 0).sum()),
        }
    reference_violations = int((parse_dates(snapshots.yoy_reference_filing_date, allow_missing=True)
                                > parse_dates(snapshots.filing_date)).sum())
    summary = {
        "ticker": "NVDA", "stage4_input": str(fundamentals_path.resolve()),
        "stage5_input": str(research_path.resolve()),
        "market_rows": len(research), "final_feature_rows": len(daily),
        "quarterly_fundamental_snapshots": int(snapshots.data_period_type.eq("quarterly").sum()),
        "annual_fundamental_snapshots": int(snapshots.data_period_type.eq("annual").sum()),
        "balance_sheet_eligible_snapshots": int(snapshots.loc[:, BALANCE_FIELDS].notna().any(axis=1).sum()),
        "selected_daily_snapshots": int(snapshots.selected_for_daily.sum()),
        "excluded_ytd_observations": int(fundamentals.data_period_type.eq("ytd").sum()),
        "quarterly_ratio_features": len(QUARTERLY_RATIO_FEATURES),
        "annual_ratio_features": len(ANNUAL_RATIO_FEATURES),
        "growth_features": len(QUARTERLY_GROWTH_FEATURES) + len(ANNUAL_GROWTH_FEATURES),
        "balance_sheet_features": len(BALANCE_RATIOS),
        "new_feature_count": len(FEATURE_COLUMNS),
        "inherited_market_feature_columns": list(WARMUP_ROWS),
        "engineered_feature_count": len(FEATURE_COLUMNS) + len(WARMUP_ROWS),
        "final_matrix_columns": len(daily.columns),
        "rows_with_quarterly_features": int(daily.loc[:, [*QUARTERLY_RATIO_FEATURES, *QUARTERLY_GROWTH_FEATURES]].notna().any(axis=1).sum()),
        "rows_with_annual_features": int(daily.loc[:, [*ANNUAL_RATIO_FEATURES, *ANNUAL_GROWTH_FEATURES]].notna().any(axis=1).sum()),
        "rows_with_q_state": streams["q"]["rows_with_state"],
        "rows_with_fy_state": streams["fy"]["rows_with_state"],
        "rows_with_both_states": int((daily.q_accession.notna() & daily.fy_accession.notna()).sum()),
        "state_streams": streams,
        "snapshot_sample_truncated_count": int(snapshots.effective_date_is_sample_truncated.sum()),
        "selected_sample_truncated_count": int((snapshots.effective_date_is_sample_truncated & snapshots.selected_for_daily).sum()),
        "date_normalization": {"parser": "Per-value ISO YYYY-MM-DD / YYYYMMDD or naive date object; comparisons use datetime64[ns]",
                               "canonical_output": "YYYY-MM-DD", "normalized_input_values": normalization, "parse_failures": 0},
        "reference_cutoff_violation_count": reference_violations,
        "historical_comparative_value_differences": diagnostic,
        "provenance_comparison": {"raw_SEC_values": "exact numeric equality; dtype differences and paired NaNs permitted",
                                  "metadata": "exact; dates compared after explicit parsing",
                                  "derived_rtol": DERIVED_RTOL, "derived_atol": DERIVED_ATOL,
                                  "market_and_stage5_preserved": validation["checks"].get("stage5_columns_preserved", False)},
        "input_sha256": hashes,
        "input_hashes_unchanged": all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.resolve())] for p in input_paths),
        "stage1_5_artifact_integrity": {
            "artifact_count": len(artifact_hashes), "before_sha256": artifact_hashes,
            "all_hashes_unchanged": all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in artifact_hashes.items()),
            "stage5_reproduction": stage5_reproduction,
        },
        "infinity_count": int(np.isinf(daily.select_dtypes("number").to_numpy(dtype=float, na_value=np.nan)).sum()),
        "missing_value_summary": {
            "market_features": daily.loc[:, list(WARMUP_ROWS)].isna().sum().to_dict(),
            "daily_features": daily.loc[:, FEATURE_COLUMNS].isna().sum().to_dict(),
            "snapshot_features": snapshots.loc[:, FEATURE_COLUMNS].isna().sum().to_dict(),
        },
        "all_missing_daily_features": [c for c in FEATURE_COLUMNS if daily[c].isna().all()],
        "comparison_status_counts": snapshots.yoy_comparison_status.value_counts().to_dict(),
        "period_matching_issues": issues.loc[:, [*METADATA_COLUMNS, "yoy_comparison_status"]].to_dict(orient="records"),
        "first_feature_date": str(daily.date.iloc[0]), "last_feature_date": str(daily.date.iloc[-1]),
        "calendar_basis": "Stage 5 observed sessions; pre-window effective dates are left-censored",
        "growth_vintage": "Latest comparable historical observation retained in the CURRENT Stage 4 processed dataset with reference filing <= current filing; frozen per snapshot. Not all SEC comparative vintages.",
        "state_policy": "Independent whole quarterly and annual states; BS chooses latest quarterly/annual snapshot with at least one BS value. No per-metric filling.",
        "state_tie_break": "Greatest filing date, period end, quarterly preference (BS/global), amendment preference, period start, lexical accession; deterministic, not intraday order.",
        "latest_filing_limitation": "A late amendment to an older period may move period_end backward; latest filing is not latest financial period.",
        "effective_age_semantics": "days_since_filing uses actual filing date; days_since_effective_date uses sample-effective date, including pre-sample truncation. No pre-window calendar is inferred.",
        "period_guard_limitation": "330-400 start/end day gaps and nonoverlap are conservative guards, not a full fiscal calendar certification.",
        "validation": validation,
        **{name: str(path.resolve()) for name, path in paths.items()},
    }
    if not summary["input_hashes_unchanged"] or not summary["stage1_5_artifact_integrity"]["all_hashes_unchanged"]:
        raise ValueError("Protected input artifact hashes changed during Stage 6 processing")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths["fundamental_feature_snapshot_path"].write_text(snapshot_csv, encoding="utf-8")
    paths["final_feature_matrix_path"].write_text(daily_csv, encoding="utf-8")
    paths["validation_path"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fundamentals-path", type=Path, default=DEFAULT_FUNDAMENTALS)
    parser.add_argument("--research-path", type=Path, default=DEFAULT_RESEARCH)
    parser.add_argument("--output-dir", type=Path, default=RESEARCH_DATA_DIR)
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW, help="Archived local SEC facts for diagnostics only")
    args = parser.parse_args()
    summary = run_pipeline(args.fundamentals_path, args.research_path, args.output_dir, args.raw_path)
    # Full period diagnostics remain in the audit JSON; the terminal summary stays compact.
    print(json.dumps({k: v for k, v in summary.items() if k not in {"period_matching_issues", "historical_comparative_value_differences"}}, indent=2))
    print(f"Period matching issues: {len(summary['period_matching_issues'])}; details in validation JSON")


if __name__ == "__main__":
    main()
