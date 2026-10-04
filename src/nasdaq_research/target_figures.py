"""Static Stage 8 sanity checks of continuous targets alone."""

import os
from pathlib import Path
import tempfile

import pandas as pd

from nasdaq_research.dates import parse_dates


def generate_target_figures(targets: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Draw three Y histograms and the primary Y series deterministically offline."""
    from nasdaq_research.targets import RETURN_COLUMNS, PRIMARY_TARGET

    previous = os.environ.get("MPLCONFIGDIR")
    with tempfile.TemporaryDirectory(prefix="nasdaq-stage8-mpl-") as config:
        os.environ["MPLCONFIGDIR"] = config
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.dates as mdates
            import matplotlib.pyplot as plt

            paths = []
            output_dir.mkdir(parents=True, exist_ok=True)
            with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                                 "figure.dpi": 100, "savefig.dpi": 120}):
                for name in RETURN_COLUMNS:
                    values = targets[name].dropna()
                    fig, ax = plt.subplots(figsize=(7, 4))
                    if len(values):
                        ax.hist(values, bins="auto", color="#376996", edgecolor="white")
                    else:
                        ax.text(.5, .5, "No available targets", transform=ax.transAxes, ha="center")
                    ax.set_title(f"NVDA {name} — next-session-open entry, n={len(values)}")
                    ax.set_xlabel("Forward return (fraction)")
                    ax.set_ylabel("Target count (overlapping horizons are not independent)")
                    path = output_dir / f"{name}_distribution.png"
                    fig.savefig(path, bbox_inches="tight", metadata={"Software": "nasdaq-research Stage 8"})
                    plt.close(fig)
                    paths.append(path)
                fig, ax = plt.subplots(figsize=(10, 3.8))
                ax.plot(parse_dates(targets.date), targets[PRIMARY_TARGET], color="#376996", linewidth=1)
                ax.set_title(f"NVDA {PRIMARY_TARGET} — realized label, not a prediction")
                ax.set_xlabel("Feature date (prediction occurs after this session closes)")
                ax.set_ylabel("Forward return (fraction)")
                locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
                ax.grid(alpha=.2)
                path = output_dir / f"{PRIMARY_TARGET}_time_series.png"
                fig.savefig(path, bbox_inches="tight", metadata={"Software": "nasdaq-research Stage 8"})
                plt.close(fig)
                paths.append(path)
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous
    return paths
