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
