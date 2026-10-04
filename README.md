# Nasdaq Research

A lightweight Python project for quantitative research on Nasdaq-related market data.

The first phase only establishes the project skeleton and local development conventions. Data ingestion, feature engineering, strategy research, backtesting, and reporting will be added in later phases.

## Project Structure

```text
nasdaq-research/
├── config/                 # Configuration templates and runtime settings
├── data/                   # Local datasets, ignored by Git except placeholders
│   ├── external/           # Third-party or reference data
│   ├── interim/            # Intermediate transformed data
│   ├── processed/          # Clean research-ready datasets
│   └── raw/                # Original downloaded or imported data
├── src/
│   └── nasdaq_research/    # Python package for research code
├── tests/                  # Test suite
├── .env.example            # Environment variable template
├── .gitignore              # Git ignore rules
├── README.md               # Project overview
└── requirements.txt        # Minimal Python dependencies
```

## Environment

Use the existing virtual environment in this workspace:

```bash
source .venv/bin/activate
python --version
```

Install dependencies only when needed:

```bash
python -m pip install -r requirements.txt
```

Create a local `.env` from the template before adding secrets or machine-specific paths:

```bash
cp .env.example .env
```

Do not commit `.env` or local data files.

## Historical Data

The initial stock pool is configured in `src/nasdaq_research/config.py`.

Download the configured one-year daily history files with:

```bash
PYTHONPATH=src python -m nasdaq_research.download
```

Each symbol is saved as a separate CSV under `data/raw/`. Local CSV data files are ignored by Git.

## Development Phases

1. Project skeleton and baseline configuration.
2. Data source selection and ingestion interfaces.
3. Data validation, cleaning, and storage conventions.
4. Feature engineering and research notebooks or scripts.
5. Strategy prototypes and backtesting workflow.
6. Reporting, experiment tracking, and reproducibility checks.

## Basic Research Features (Stage 3)

Process the existing raw CSV files offline:

```bash
PYTHONPATH=src ./.venv/bin/python -m nasdaq_research.process
```

Each symbol is saved under `data/processed/`, which remains ignored by Git.
The original date and OHLCV columns are preserved. Features include simple and
log returns, intraday return, daily range, 5/20/60-day SMAs, and 20/60-day
rolling return volatility (sample standard deviation, ddof=1, daily, unannualized).
All windows use the current and preceding rows only. These are end-of-day
features, available after the current day's OHLCV is known.

Warm-up NaNs remain empty in CSV: returns need one preceding close; SMAs need
4/19/59 preceding rows; volatility needs 20/60 preceding rows because the first
return is missing. Validation distinguishes these expected NaNs from unexpected
missing or infinite values. Invalid raw data, duplicate dates, and unordered
dates are rejected without dropping, filling, or reordering rows.

Run the full offline test suite:

```bash
PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests -v
```

## NVDA Fundamentals (Stage 4)

The pipeline uses the official SEC Company Facts endpoint:
<https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json>.
It maps US-GAAP facts into income statement, balance sheet, and cash flow fields.
No valuation, forecasting, or trading logic is included; no new dependencies are needed.

Run from the repository root:

```bash
PYTHONPATH=src ./.venv/bin/python -m nasdaq_research.fundamentals
```

Requests carry an identifying research User-Agent. Set `SEC_USER_AGENT` to your
own application name and real contact information for ongoing use; the default
identifies this research application without inventing contact details.

Outputs (ignored by Git):

- `data/fundamentals/raw/NVDA_companyfacts.json`: exact downloaded response bytes.
- `data/fundamentals/processed/NVDA_fundamentals.csv`: normalized observations.
- `data/fundamentals/processed/NVDA_validation.json`: quality checks, missing
  counts, exclusions, duplicate source fact counts, and normalization issues.

Reproduce the processed files from the archived response without network:

```bash
PYTHONPATH=src ./.venv/bin/python -m nasdaq_research.fundamentals \
  --raw-input data/fundamentals/raw/NVDA_companyfacts.json
```

`fundamental_mapping.py` centrally defines ordered candidates, units, and
instant/duration basis. Values use USD, except diluted EPS in USD/share.
Each mapped field retains a `_concept` provenance column. Unavailable fields
are empty; contradictory values for the preferred concept are reported and
left empty. FCF is OCF minus positive PP&E CapEx only when both are present for
the identical accession and duration. No missing values are imputed.

`data_period_type` distinguishes `annual` (330–400 days, 10-K), `quarterly`
(70–110 days), and `ytd` (111–300 days, Q2/Q3 10-Q). Unsupported durations are
reported and excluded. No standalone cash-flow quarters are inferred by
subtracting cumulative values. Q1 cumulative values span one quarter and can
be classified quarterly. A single filing can therefore have separate quarterly
and YTD rows. Balance sheet fields are end-of-period snapshots repeated on
those rows, not flows over their durations.

Records retain fiscal year/period, start/end, filing date, form, and accession.
Only the latest end reported in each accession is retained: comparative facts
have the filing's fiscal metadata and cannot safely be treated as earlier
fiscal-year observations. Different accessions, including amendments, remain
separate; this is a filing history rather than a latest-restated period table.
Fiscal years follow NVIDIA's labels, not calendar-year end dates. Validation
checks metadata, period duration, finite numeric-or-missing values, observation
uniqueness, and FCF consistency. Use filing dates when deciding availability;
these outputs alone do not constitute a point-in-time investment dataset.

Known limitations: Company Facts excludes custom/dimensional concepts
([SEC API documentation](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)).
Mapping deliberately excludes `PaymentsToAcquireProductiveAssets`, which
combines PP&E, software, and intangibles, rather than silently changing the
PP&E CapEx definition. As a result, many periods, including recent ones, lack
CapEx and FCF. `long_term_debt` is noncurrent debt only; broader debt concepts
are not substituted. Some periods lack directly reported total liabilities.
EPS is as disclosed in each filing and can change with stock splits or
restatements; it is not retrospectively harmonized. Raw API snapshots are
replaced on each download; archive them separately to preserve multiple vintages.

## Point-in-Time Research Dataset (Stage 5)

Read existing Stage 3/4 CSVs offline, without downloading or rebuilding them:

```bash
PYTHONPATH=src ./.venv/bin/python -m nasdaq_research.alignment
```

Outputs are `data/research/NVDA_research.csv` and `NVDA_validation.json`.
All Stage 3 columns and their warm-up missing values are retained; `ticker=NVDA`
is added because Stage 3 uses per-symbol files. Financial values and concept
provenance columns are copied from one whole observation, with metadata renamed
`fundamental_*`. `days_since_filing` and `days_since_effective_date` count calendar
days. No field-level filling or backward filling occurs.

Availability is strictly after filing day. The effective date is the next date
present in the market dataset, and a backward as-of join uses effective dates,
never period ends. This uses observed trading sessions rather than guessed
business weekdays; weekends and exchange holidays absent from the input are
skipped. The market input must contain all trading sessions over its coverage.
For filings before coverage begins, effective dates are **left-censored to the
first observed session**: they describe first availability within this dataset,
not the historical exchange session immediately following that old filing.
A filing after the final session is never matched. The report identifies this
calendar convention. An extended exchange calendar would be needed to recover
exact pre-window effective dates.

Only standalone `quarterly` and `annual` rows are eligible. YTD is excluded,
including its cash-flow values; missing quarterly cash flow remains missing.
For competing observations on one effective date, select the greatest tuple:
filing date, period end, quarterly preference over annual, amendment preference,
period start, lexical accession. The final accession tie-break is deterministic,
not a claim about intraday filing order. Selection preserves complete source
rows. A later amendment changes data only after its own filing date; older
snapshots are never rewritten. Different annual and quarterly durations remain
explicit in `fundamental_data_period_type`; they are not directly comparable
flows and this stage does not derive TTM or annualized features.

Validation checks every market row/column and every attached value/provenance
cell against the selected source observation, plus strict filing-day exclusion,
effective date bounds, ticker/order/uniqueness and information age. Offline
regressions cover pre-filing missing data, filing day, next session, holidays,
amendments, deterministic ties, YTD exclusion, future mutations and prefix
invariance. The current archived input has no amended or repeated-period rows;
synthetic tests cover those scenarios. This alignment inherits the archived
Company Facts vintage and Stage 4 provenance limitations; it cannot establish
that SEC never revised the upstream historical facts.

## Research Feature Engineering (Stage 6 / 6.1)

Run independently using existing Stage 4/5 CSVs, entirely offline:

```bash
PYTHONPATH=src ./.venv/bin/python -m nasdaq_research.research_features
```

Optional arguments are `--fundamentals-path`, `--research-path`, `--raw-path`
(local archived facts for diagnostics only), and `--output-dir`. Outputs under `data/research/` are
`NVDA_fundamental_features.csv` (all eligible filing-level snapshots),
`NVDA_features.csv` (Stage 5 columns plus features and comparison provenance),
and `NVDA_feature_validation.json` (validation, counts, missing values and
period-matching and comparative-value diagnostics). Earlier artifacts are read
only; the report records hashes and verifies Stage 5 reproduction in memory.
Calendar dates accept ISO `YYYY-MM-DD`, `YYYYMMDD`, or naive date objects.
Invalid/ambiguous dates, numeric epochs and intraday/timezone timestamps are
rejected. Comparisons and sorting use `datetime64[ns]`; CSV dates use `YYYY-MM-DD`.

There are 30 new features: seven `q_*` ratios and seven `fy_*` ratios
(`gross_margin`, `operating_margin`, `net_margin`, `rd_to_revenue`,
`sga_to_revenue`, `ocf_margin`, `fcf_margin`), four balance-sheet ratios
(`current_ratio`, `cash_to_assets`, `liabilities_to_assets`, `debt_to_assets`),
and six YoY growth features per basis (`revenue`, `gross_profit`,
`operating_income`, `net_income`, `rd`, `operating_cash_flow`). Income/cash-flow
ratios divide the corresponding field by revenue **within one observation**.
Balance ratios use current assets/current liabilities, cash/assets,
liabilities/assets and noncurrent long-term debt/assets, respectively.

Quarterly and annual flows remain separate. Quarterly YoY is current value /
previous FY's same fiscal quarter value − 1; annual YoY uses previous FY's
annual value. A reference must have been filed on or before the current
snapshot's filing date, with an explicit cutoff check. Select the latest
comparable observation **retained in the current Stage 4 processed dataset**,
ordered by filing date, period end, amendment preference, period start and
lexical accession. Freeze this reference at the
current filing; later reference amendments do not rewrite past snapshots.
`yoy_reference_*` columns retain its full period/filing identity. A fiscal key
with multiple distinct periods known at that time is ambiguous. Require both
start and end dates to be 330–400 days apart, with nonoverlapping periods;
otherwise growth is missing and `yoy_comparison_status` explains why. This
conservative guard catches inconsistent early source labels without relabeling
Stage 4 or choosing adjacent quarters. The 330–400-day guard admits 52/53-week
years but does not certify a fiscal calendar or repair calendar changes.
Stage 4 excludes comparative facts outside each accession's latest end date;
its retained references are not necessarily the latest of all public SEC
versions. Archived same-concept/unit/physical-period redisclosures are compared
separately as **historical comparative value differences**, without changing
denominators or inferring that every difference is a restatement.

Ratios and growth require a positive denominator. Missing inputs, zero or
negative denominators, absent/ambiguous comparisons and arithmetic overflow
remain NaN. Negative current earnings with a positive comparison denominator
use the stated formula. No YTD flows, subtraction to infer quarters, field
filling, interpolation, clipping, scaling or imputation are introduced. Growth
is computed on filing-level history: daily repeated financial values cannot
be fed to `pct_change()` to recover financial-period growth.

Daily mapping reuses Stage 5's effective-date helper and whole-observation
as-of selection, preserving its filing-day exclusion and observed-calendar
coverage limitation. Snapshot `effective_date` uses that same calendar;
`selected_for_daily` identifies the snapshots actually selected in the market
window across any state. `selected_for_q`, `selected_for_fy`, `selected_for_bs`
and `selected_for_stage5` identify each selector. Historical snapshots remain
in the file for inspection. Independent whole quarterly and annual streams
carry their features until that same basis receives a later effective filing;
one basis never clears the other. Metadata is `q_*` and `fy_*` (filing/effective
dates, accession, form, fiscal year/period, physical period, basis and ages),
with separate `q_yoy_reference_*` and `fy_yoy_reference_*` provenance. Existing
Stage 5 fields and unprefixed `yoy_reference_*` retain the single-observation
checkpoint context; they do not identify both independent states.

The balance-sheet stream selects the latest quarterly/annual observation with
at least one of its seven source BS fields present. All four ratios, `bs_*`
metadata and seven `bs_` source values come from that same whole snapshot.
Missing metrics stay missing; income-only observations do not replace an
available BS state. Each stream uses Stage 5's documented deterministic tie
order. State means **latest filing**, not latest financial period: an older
period's late amendment may move `period_end` backward. The report counts these
transitions; no previous daily row is rewritten.

Snapshot and state `effective_date_is_sample_truncated`/`effective_date_basis`
explicitly identify filings before the first market session. For those rows,
`days_since_effective_date` measures time since the **sample-effective** date,
not the unavailable historical next exchange session. `days_since_filing`
always uses the real filing date. No pre-window trading calendar is guessed.

All nine Stage 3 market features and their warm-up NaNs are retained. SMAs
remain trailing close means over 5/20/60 sessions; volatility remains the
trailing 20/60-session standard deviation of simple returns, `ddof=1`, without
annualization. Stage 3 has no momentum or price/MA-ratio feature, so none is
added here. Validation audits every Stage 5 cell and derived snapshot against
source history. Identifiers/metadata and raw SEC values use exact equality
(numeric dtype changes and paired NaNs allowed); derived ratios/growth use
explicit `rtol=0`, `atol=1e-12`. Original market columns are independently
checked for preservation, including temporary-name collisions. Adversarial
tests include a small independent daily-loop oracle without production joins.
CSV export preserves original Stage 4/5 source-cell text (canonicalizing dates)
and validates a reload before saving, avoiding decimal-parser precision drift.
This stage includes no ROA/ROE, TTM, valuation, interactions,
prediction or strategy features. Run all offline regression tests with the
existing `unittest discover` command above.

## Feature Diagnostics & Exploratory Research (Stage 7)

Diagnose the existing Stage 6.1 outputs offline, without downloading or rebuilding
price/SEC data:

```bash
PYTHONPATH=src ./.venv/bin/python -m nasdaq_research.diagnostics
```

Optional arguments: `--features-path`, `--snapshots-path`, `--validation-path`,
and `--output-dir`. The Stage 6.1 validation must pass. The command reads
`NVDA_features.csv`, `NVDA_fundamental_features.csv`, and
`NVDA_feature_validation.json`; it reports the actual shape, ticker and dates.
SHA-256 checks protect the input files and existing Stage 1–6.1 data, including
`NVDA_research.csv`. All outputs go to `data/research/diagnostics/`.

Feature classification reuses the Stage 3/6 feature definitions. The full column
inventory distinguishes market, quarterly, annual and balance-sheet research
features from numeric metadata (ages/fiscal years), provenance and raw source
values. Only research features enter distributions and correlations.

Diagnostics include non-null coverage, missing blocks and longest missing runs;
sample standard deviation, quantiles, skewness and excess kurtosis; transparent
outer IQR fences (`Q1 − 3×IQR`, `Q3 + 3×IQR`); near-constant values; market
warm-up checks and unusually long constant runs; and fundamental changes against
each stream's effective date. Missing-to-value and value-to-missing changes are
checked too. Timing inconsistencies are reported without changing source data.

Thresholds are centralized in `DiagnosticThresholds`: 100% coverage is full,
95–100% is high, 50–95% is moderate, below 50% is sparse, and zero is all missing.
Near-constant means one distinct finite value with at least two observations,
or a dominant value of at least 99% with at least 20 observations. Outliers
require at least eight finite observations and positive IQR; zero-IQR and small
samples remain explicitly unassessed. Short history means fewer than 60 finite
daily observations or five finite window states.

Pearson correlation uses finite pairwise complete observations and records every
pair's sample count. At least 20 complete observations are required for a
coefficient in any view; `abs(correlation) >= 0.95` flags potential redundancy.
All-missing and constant features remain in reports/matrices with undefined
coefficients. No feature is deleted. Daily fundamental pairs also report each
feature's distinct filing-state count within that pair's overlap.

Fundamental distributions/correlations have separate `daily_repeated`,
`window_snapshot` and `historical_snapshot` views. Window states are identified
by whole filing provenance, never by unique feature values. Near-constant,
outlier and short-history quality flags for fundamentals use these window
states, so normal filing-driven repetition is not treated as a defect.
Historical statistics read the archived snapshot file directly, within each
basis separately, using filings strictly before the last observed session.
They describe a longer history than the daily window and inherit its archived
vintage and source limitations. Small window-state samples produce counts and
distributions but no correlation coefficients under the minimum-sample rule.

Outputs:

- `feature_inventory.csv`, `feature_missingness.csv`, `feature_quality.csv`:
  column classification, coverage and diagnostic flags/notes.
- `feature_distributions.csv`: daily, window-state and historical statistics.
- `feature_correlations.csv`, `correlation_pairwise_counts.csv`,
  `correlation_pairs.csv`, `high_correlation_pairs.csv`: full daily matrix,
  overlap counts and pair diagnostics.
- `temporal_diagnostics.csv`, `fundamental_states.csv`,
  `snapshot_correlations.csv`: timing checks, observed state repetition and
  within-basis snapshot pair diagnostics.
- `NVDA_feature_diagnostics.json`: input hashes, thresholds, counts, known
  missingness, limitations, validation and output hashes.
- `figures/*.png`: coverage, four group correlation heatmaps, existing 20/60-day
  volatility series and quarterly revenue-growth/gross-margin step plots.

Matplotlib is the only added direct dependency; figures use its offline Agg
backend with no dashboard. Reports omit run timestamps and use no random steps;
tests compare both reports and PNG bytes across repeated runs in one environment.
Run Stage 7 tests with `unittest discover -s tests -p 'test_diagnostics.py' -v`,
or use the full offline suite command above for all stages.

`usable` means data quality permits further research; it says nothing about
predictive ability or model suitability. The current daily sample is about one
year, with far fewer distinct fundamental states. Daily repetitions are not
independent fundamental observations; even distinct filings can be dependent.
Correlation establishes neither causality nor predictive power, and these
results describe only NVDA's current window. This stage has no prediction target
or future-return analysis: target design belongs to Stage 8. Stage 7 performs
no feature selection, cleaning, imputation, winsorization, scaling, interaction
engineering, modeling, valuation or backtesting.

## Target Definition & Label Engineering (Stage 8)

Construct continuous supervised-learning labels from the existing feature
matrix, entirely offline:

```bash
PYTHONPATH=src ./.venv/bin/python -m nasdaq_research.targets
```

`X_t` is available **after trading session t closes**, because market features
use that day's full OHLC and trailing data. The earliest assumed entry is
therefore **the next observed session's open**, `Open[t+1]`. Using `Close[t]`
as entry would assume execution at a price already used to construct X.

| Target | Definition | Exit | Role |
|---|---|---|---|
| `forward_return_1d` | `Close[t+1] / Open[t+1] - 1` | Next session close | Secondary |
| `forward_return_5d` | `Close[t+5] / Open[t+1] - 1` | Fifth subsequent session close | **Primary** |
| `forward_return_20d` | `Close[t+20] / Open[t+1] - 1` | Twentieth subsequent session close | Secondary |

Horizons count positions in the existing observed market rows, never calendar
days. Weekends and absent holiday sessions are skipped without generating
dates or downloading a calendar. The pipeline explicitly supports only NVDA,
rejects duplicate/unordered dates, and requires finite, positive numeric open
and close prices. Invalid prices cause failure without substitution or filling.

Outputs:

- `data/research/targets/NVDA_targets.csv`: one row per feature date, containing
  `date`, `ticker`, entry date/open, and each horizon's exit date/close and return.
  `date` is the feature date; it is not duplicated as another provenance column.
- `data/research/NVDA_labeled.csv`: all original feature columns in original
  order plus eleven target/provenance columns. All rows, metadata and NaNs remain.
- `data/research/targets/NVDA_target_validation.json`: actual counts, strict
  formula/date/position/source-price checks, source/output hashes and limitations.
- `data/research/targets/target_summary.csv`: only target distributions, including
  counts, missingness, mean, sample std, quantiles and extrema.
- `data/research/targets/figures/`: three target histograms and the primary target
  time series, without feature-target comparisons or prediction lines.

Optional flags: `--features-path`, `--output-dir`, `--labeled-path`,
`--feature-validation-path`, and `--diagnostics-path`. Successful Stage 6.1/7
reports must describe the input, including the Stage 7 checkpoint's feature
hash. The labeled file must retain the name `NVDA_labeled.csv` to protect older
artifacts. SHA-256 checks cover existing Stage 1–7 data, including all Stage 7
diagnostic outputs. The source feature file is never rewritten. CSV export
preserves original X and future price decimal tokens, and validates a reload;
source prices are compared exactly and derived returns use `rtol=0`, `atol=1e-12`.

Targets intentionally use future prices; features cannot. Future information is
confined to the appended Y/provenance columns, and every original X cell is
validated for preservation. Rows without enough future sessions keep NaN labels:
the missing counts are computed from actual data, including samples shorter
than a horizon. Available entry provenance is retained even when a later exit
is unavailable. No rows are dropped and no targets are transformed or clipped.

Adjacent 5/20-session intervals overlap substantially. Daily label counts are
not independent sample counts, especially for 20-session labels. Entry/exit
dates are retained for later chronological split, purge and embargo design.
Later modeling must not use a simple shuffled random split such as
`train_test_split(..., shuffle=True)`. Stage 8 implements no split, purge or
embargo; these belong to Stage 9. The current window is about one year, prices
inherit the existing archived OHLC basis, and transaction costs/slippage are
not included. Observed prices do not guarantee actual execution.

This stage defines no model, classification/threshold labels, feature selection,
feature-target correlations, information coefficients, strategy or backtest.
Run the Stage 8 offline tests, including an independent scalar oracle, with:

```bash
PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests -p 'test_targets.py' -v
```

Use the full `unittest discover -s tests -v` command above for regression checks.
Repeated runs produce identical CSV/JSON/PNG artifacts in the same environment.

## Stage 9 — Leakage-Safe Temporal Splitting & Evaluation Protocol

Create manifests from the existing Stage 8 labeled dataset, without rebuilding
upstream files or training any model:

```bash
PYTHONPATH=src ./.venv/bin/python -m nasdaq_research.splits
```

The primary target remains `forward_return_5d`. The last **50 primary-labelable
observations** form the locked final test; all earlier primary-labelable rows
form Development. Primary missing labels must be a contiguous unlabeled tail,
retained and excluded from every target's training/evaluation, even when a
secondary label is available. No feature rows or columns are deleted.

With Stage 9.1, the last 75 Development observations whose labels for all
registered horizons have matured strictly before final-test start form
**three chronological validation blocks of 25 rows**. Each fold's nominal training candidates are all preceding
Development observations, so earlier validation periods can become later
training candidates. Configuration lives in `SplitConfig`; the default requires
at least 60 nominal initial training observations, without shrinking the fixed
50/25/3 protocol to fit insufficient data. Each effective training pool must
also be nonempty. Unordered dates, duplicate dates, mixed tickers, missing
required provenance and interior primary-label gaps are rejected.

For every fold and the final training pool, a training candidate is usable only
when its own target exists and **`target_exit_date < evaluation_start_date`**.
Equality is purged. This uses actual Stage 8 exit provenance, never a fixed
number of trailing rows. The 1d/5d/20d targets share the same nominal validation
and final-test date blocks but receive independent training purges. Missing
secondary labels are marked `target_missing` with `is_usable=false` in training
and the final test. CV validation requires complete labels for every registered
horizon, so every validation block remains fully usable for all three targets.

`embargo_sessions=0`: training is always before evaluation and its overlapping
labels are purged. This causal expanding-window protocol adds no post-validation
embargo. Nonzero embargo is explicitly unsupported rather than silently ignored;
another CV architecture would require a new decision. Random splits, shuffled
K-fold and `train_test_split(..., shuffle=True)` are prohibited.

**Final test is a one-shot out-of-sample evaluation set.** It is locked to the
source hash, configuration, dates and evaluation contract. Rerunning cannot
silently move its boundary or replace the registered primary metric. Final-test
performance must never choose targets, algorithms, features, hyperparameters
or preprocessing; it has not been evaluated in Stage 9. Manifest `is_usable`
denotes data eligibility, not authorization to use locked test results during
development. Roles are scoped by `(fold, target_name, date)`; `not_candidate`
keeps future rows visible while excluding them from a CV fold.

Stage 10's preregistered regression contract is **MAE primary**, with RMSE, R²,
Pearson correlation, Spearman correlation and directional accuracy secondary.
The primary metric cannot be changed after seeing results. CV MAE is aggregated
with equal fold weights. Baselines are zero return and historical mean; the
latter may use only that horizon's usable, purged training-fold targets, never
validation, test or full-sample means. Baselines and metrics are defined but
not executed here. Any later imputation, scaling, normalization, feature
selection or target-dependent filtering must fit on usable training data only,
then transform validation/test. Full-sample preprocessing is prohibited.

Outputs under `data/research/splits/`:

- `NVDA_split_manifest.csv`: all dates × horizons × CV/final contexts, with
  nominal membership, usable/purged/missing/tail roles, entry/exit dates, locked
  test flags, a `pre_test_gap` flag and role, and explicit final-training
  candidate/purged/usable flags.
- `NVDA_cv_folds.csv`, `NVDA_final_training_pool.csv`: nominal/effective training
  counts, horizon-specific purge counts, evaluation dates/completeness, maximum
  validation exit dates, and gap rows returning to final training.
- `NVDA_split_summary.csv`: primary partition sizes, gap dates/counts,
  CV history/validation sizes and per-horizon counts.
- `NVDA_split_validation.json`: independent role/date audits, source/output
  hashes, reproducibility metadata and readiness.
- `NVDA_evaluation_protocol.json`: fixed split, purge, metrics, baselines,
  preprocessing restrictions, lock and limitations for later stages.

CLI path overrides are `--labeled-path`, `--targets-path`, `--features-path`,
`--target-validation-path`, and `--output-dir`. Stage 8 hashes and formulas are
audited read-only, then all existing Stage 1–8 data hashes are checked again.
No train/validation/test copies, figures or new dependencies are needed.
Run Stage 9 tests with `unittest discover -s tests -p 'test_splits.py' -v`, then
the full suite command above. CSV/JSON bytes are reproducible across repeats.

The current window is about one year; the 50-row test and 25-row validation
blocks contain fewer independent observations because labels overlap. Purge
reduces training sizes further, especially at 20d. No model, transaction-cost
simulation, feature selection or preprocessing is performed.

### Stage 9.1 — Validation/Test Boundary Hardening

Feature-date separation alone does not isolate the test: a validation feature
date can precede test start while its forward label still needs prices inside
the test period. The original Stage 9 Fold 3 had 1/5/20 such validation labels
for 1d/5d/20d. Stage 9.1 requires every model-selection validation label to
satisfy **`target_exit_date < final_test_start_date`**; equality is unsafe.

`pre_test_gap` is the Development suffix after the latest observation with
complete labels and pre-test exits for **all registered horizons** (1d, 5d,
20d). Its dates and row count are derived from actual provenance, not a fixed
20-row offset or calendar-day subtraction. The longest registered horizon is
currently 20d. Missing labels or exit metadata never count as safe. Any interior
ineligible observations are also excluded from the eligible CV validation
history; they remain present in the manifest and can still be training
candidates subject to the existing purge.

The last 75 eligible observations form three full 25-row validation blocks,
moving the complete CV schedule earlier. Initial nominal training remains at
least 60 rows; insufficient post-gap history fails explicitly without changing
the 50-row test, 25-row folds or three-fold count. Every fold, not only Fold 3,
is audited for complete pre-test validation labels. The validation JSON reports
per-horizon crossing counts, maximum validation exits, gap dates/counts and
all three leakage boundaries.

Gap rows receive `role=pre_test_gap` and `is_usable=false` in CV contexts. They
are **not permanently discarded**: in the `final` context they are ordinary
Development candidates, usable or purged independently for each target using
its own `target_exit_date < final_test_start_date`. The global `pre_test_gap`
flag keeps their membership visible. This separates model-selection eligibility
from final-training eligibility.

Training purge and the pre-test gap protect different boundaries:

- Training → Validation: each training label exits before its fold's validation start.
- Validation → Final Test: all registered validation labels exit before test start.
- Final Training → Final Test: each final-training label exits before test start.

The gap is distinct from embargo; `embargo_sessions` remains zero. Final test
remains the last 50 primary-labelable rows, currently 2026-07-15 through
2026-09-23, and the five-row unlabeled tail remains outside all evaluations.
MAE, secondary metrics, baselines and training-only preprocessing/selection
contracts are unchanged. Gap design uses label availability only, without
inspecting target distributions or performance. No modeling is performed.

The same CLI upgrades a recognized, uncommitted Stage 9 protocol to version
`9.1` only after checking its source hash, original dates and unchanged final-test
and evaluation contracts. A version 9.1 protocol locks the new CV/gap boundaries
as well; subsequent runs cannot silently move them. CSV/JSON repeats remain
byte-for-byte reproducible. The gap further reduces CV training history, and
overlapping 20d labels still provide limited independent information.

### Stage 10 — Leakage-Safe Baseline Modeling

Run the development-CV benchmark locally on CPU:

```bash
PYTHONPATH=src .venv/bin/python -m nasdaq_research.modeling
```

Stage 9.1's manifest and evaluation protocol are the sole split authority.
The CLI verifies their frozen checkpoint hashes before fitting; it never
rebuilds splits. Only `forward_return_5d` is modeled, in its original return
units. Effective training counts are currently 96 / 121 / 146, with 25
validation observations per fold. The resulting wide OOF table has 75 unique,
chronological validation dates and six prediction columns. Training, pre-test
gap, final-test and unlabeled-tail predictions are excluded.

The six registered candidates are `zero_return`, `historical_mean`,
`ols_market`, `ols_all`, `ridge_market` and `ridge_all`. Historical mean uses
only the current fold's effective training targets. OLS uses
`LinearRegression`; Ridge fixes `alpha=1.0` and uses the deterministic SVD
solver. Both retain an intercept. No tuning, predictive feature selection,
extra models or final-model fit is performed. The Stage 9.1 final training
pool is checked for immutability but is not used to fit a model.

Market candidates are the nine Stage 7 market research features; All candidates
are the 39 registered research features (9 market, 13 quarterly, 13 annual,
4 balance-sheet). Inventory/classification is verified explicitly; numeric
column selection is prohibited. Targets, target provenance, raw OHLC/SEC
values and metadata cannot enter X. Stage 7 overall quality/coverage scores
do not decide fold-level feature retention.

Each OLS/Ridge fold fits its own preprocessing on effective training rows
only: retain coverage **>= 0.50**, fill missing values with training medians,
drop exactly constant imputed training features, then fit `StandardScaler`
with population variance (`ddof=0`). Validation uses those fixed medians and
scaler parameters, including newly missing values; validation never fits a
transform. Feature coverage, drop reasons, medians, means, standard deviations,
scaler scales and fit dates are saved for audit. Coefficients use standardized
feature units and remain descriptive diagnostics.

Ranking uses equal-fold **mean MAE**, with sample standard deviation
(`ddof=1`). Exact ties share rank and are reported together. Fold and pooled
OOF metrics also include RMSE, R², Pearson, Spearman and directional accuracy
using `sign`, with exact zero its own category. Negative R² is retained.
Undefined correlations for constant predictions are empty CSV values (NaN);
undefined JSON values are `null`.

Seven artifacts are written atomically under `data/research/modeling/`:

- `NVDA_baseline_oof_predictions.csv`
- `NVDA_baseline_cv_metrics.csv`
- `NVDA_baseline_cv_summary.csv`
- `NVDA_baseline_feature_usage.csv`
- `NVDA_baseline_coefficients.csv`
- `NVDA_baseline_protocol.json`
- `NVDA_baseline_validation.json`

Validation independently checks training statistics, fit dates, historical
means, saved-coefficient reconstruction of OOF predictions, metric integrity,
CSV round trips and upstream SHA-256 immutability. JSON records input/output
hashes and software versions; timestamps are omitted and BLAS uses one thread
so repeated runs produce identical bytes in this environment. Existing Stage
10 protocol changes fail explicitly. CLI overrides are `--labeled-path`,
`--split-dir`, `--diagnostic-dir` and `--output-dir`.

**Final Test remains locked:** `final_test_predictions_generated=false` and
`final_test_metrics_computed=false`. The 20-session pre-test gap is excluded
from all CV fitting, preprocessing, model selection, metrics and predictions.
The pipeline does not ingest secondary targets or raw target prices into its
modeling dataset. The full source file is hashed only for checkpoint integrity.

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_modeling.py' -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Stage 10 tests include independent one-feature OLS/Ridge closed-form oracles,
known-value metric checks, exact 50% coverage and constant-feature boundaries,
and adversarial gap, final-test feature/target, secondary-target and validation
mutations. Validation mutations test the same fold's fit: earlier validation
rows may legitimately become training rows in later expanding folds.

This is a development-CV benchmark for approximately one year of NVDA data,
with only 75 OOF observations and 25 per fold. Overlapping 5d labels mean
observations are not independent. There is no nonlinear modeling, transaction
cost calculation or backtest, and no final-test performance has been evaluated.

### Stage 10.1 — Baseline Model Stability Diagnostics

Stage 10's learned linear models have large cv_3 errors. This stage diagnoses
that instability while preserving the fixed target, features, preprocessing,
Ridge alpha, folds and Stage 10 candidate ranking. It introduces no model,
tuning, feature selection or treatment. Stage 10 code and baseline artifacts
remain unchanged.

```bash
PYTHONPATH=src .venv/bin/python -m nasdaq_research.model_stability
```

The CLI checks the Stage 9.1 and Stage 10 frozen hashes, reconstructs the same
four learned-model fits on effective training rows, and verifies saved OOF
predictions and metrics with `rtol=0, atol=1e-12`. Diagnostics cover the 75 OOF
validation observations; row errors include all six candidates (450 records).
Residuals consistently mean **actual minus prediction**.

Outputs under `data/research/modeling/stability/` include row errors,
prediction distributions, raw train-versus-validation feature shift,
training-scaler validation z-scores, per-feature prediction contributions,
coefficient drift, training design-matrix conditioning, OLS training influence,
fundamental state context, and summary/validation JSON. All candidate features
receive raw shift statistics; dropped features receive no artificial z-score
or contribution. Strict raw range comparisons exclude equality and missing
values; range fractions use non-null validation values. Descriptive standard
deviations use `ddof=1`; Stage 10 scaler parameters retain `ddof=0`.

Contributions reconstruct every learned prediction as intercept plus the sum
of coefficient times z-score. Missing-fold coefficients remain NaN. Sign flips
count opposite nonzero signs in consecutive available folds; missing folds are
skipped and exact zero is a separate sign category. Coefficient norms and
Ridge/OLS ratios are descriptive only.

Conditioning uses NumPy SVD on the actual post-imputation/scaling training X,
with the numerical rank threshold `max(shape) * eps * largest singular value`.
Numerically rank-deficient matrices have infinite condition number (JSON null
with an explicit infinity flag). Training correlation counts use unique pairs.
OLS influence uses the intercept-inclusive Moore-Penrose projection, internal
studentization and effective-rank residual degrees of freedom. Undefined
quantities retain NaN and a reason. No observation is deleted or refitted.

State context compares a validation row's selected q/fy/bs source identity with
its immediately previous observed session. Existing calendar-day filing and
sample-effective ages preserve Stage 6.1 semantics; metadata never enters X.
State associations and linear diagnostics do not establish causality.

Five figures under `stability/figures/` show actual versus prediction, prediction
ranges, cv_3 largest errors, largest validation z-scores and coefficient drift.
Each is marked **development CV only**. Feature ordering in these figures is
for visualization and cannot change the fitted feature set.

Final Test remains locked, with predictions and metrics both false. Pre-test
gap, Final Test and unlabeled-tail rows are excluded from all diagnostics.
The CLI executes in-memory feature/target/state-metadata isolation mutations,
audits CSV round trips, and checks Stage 1–10 SHA-256 immutability. Repeated
machine-readable outputs are deterministic. Path overrides are `--labeled-path`,
`--split-dir`, `--diagnostic-dir`, `--baseline-dir` and `--output-dir`;
`--no-figures` skips only plotting.

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_model_stability.py' -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Tests include independent range, z-score, two-feature OLS contribution,
conditioning and influence oracles, coefficient-sign/missing-fold checks,
artifact tampering and scope mutations. Validation mutations protect the same
fold's fit; earlier validation may legitimately enter later expanding training.
This stage explains the development benchmark only, without opening the test
or implementing any next-stage intervention.
