"""Data access and ingestion placeholders."""

from pathlib import Path


def data_path(*parts: str) -> Path:
    """Return a path inside the local data directory."""
    from nasdaq_research.config import DATA_DIR

    return DATA_DIR.joinpath(*parts)
