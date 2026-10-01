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
