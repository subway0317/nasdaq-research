"""Deterministic static Matplotlib figures for Stage 7 diagnostics."""

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from nasdaq_research.dates import parse_dates


def generate_figures(data: pd.DataFrame, tables: dict, output_dir: Path, ticker: str,
                     correlation_min_pairwise: int) -> list[Path]:
    # Local temporary config avoids writes to the user's home or persistent caches.
    import os
    previous = os.environ.get("MPLCONFIGDIR")
    with tempfile.TemporaryDirectory(prefix="nasdaq-stage7-mpl-") as config:
        os.environ["MPLCONFIGDIR"] = config
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.dates as mdates
            import matplotlib.pyplot as plt
            with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                                 "figure.dpi": 100, "savefig.dpi": 120}):
                paths = _draw(data, tables, Path(output_dir), ticker, correlation_min_pairwise, plt, mdates)
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous
    return paths


def _draw(data, tables, output_dir, ticker, correlation_min_pairwise, plt, mdates):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(fig, name):
        path = output_dir / f"{name}.png"
        fig.savefig(path, bbox_inches="tight", metadata={"Software": "nasdaq-research Stage 7"})
        plt.close(fig)
        paths.append(path)

    def date_axis(ax):
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set_xlabel("Observed trading date")
        ax.grid(alpha=.2)

    colors = {"market": "#376996", "quarterly": "#4c956c", "annual": "#bb7e5d", "balance_sheet": "#826c9c"}
    inventory = tables["feature_missingness"]
    fig, ax = plt.subplots(figsize=(10, max(5, len(inventory) * .23)))
    positions = np.arange(len(inventory))
    ax.barh(positions, inventory.coverage_ratio, color=[colors[g] for g in inventory.feature_group])
    ax.barh(positions, inventory.missing_ratio, left=inventory.coverage_ratio, color="#e2e2e2")
    ax.set_yticks(positions, inventory.feature_name, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Non-null coverage (colored) / missing fraction (gray)")
    ax.set_title(f"{ticker} research feature coverage — {len(data)} daily rows")
    save(fig, "missingness")

    matrix = tables["feature_correlations"]
    for group in colors:
        names = inventory.loc[inventory.feature_group.eq(group), "feature_name"].tolist()
        if not names:
            continue
        values = matrix.loc[names, names].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(max(6, len(names) * .62), max(5, len(names) * .56)))
        cmap = plt.get_cmap("RdBu_r").with_extremes(bad="#dedede")
        heat = ax.imshow(np.ma.masked_invalid(values), vmin=-1, vmax=1, cmap=cmap)
        ax.set_xticks(np.arange(len(names)), names, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(names)), names, fontsize=8)
        ax.set_xlabel("Research feature")
        ax.set_ylabel("Research feature")
        note = "daily repeated fundamentals" if group != "market" else "daily observations"
        ax.set_title(f"{ticker} {group}: Pearson correlation\n{note}; gray = undefined or n < {correlation_min_pairwise}")
        fig.colorbar(heat, ax=ax, label="Pearson correlation", shrink=.8)
        save(fig, f"correlation_heatmap_{group}")

    dates = parse_dates(data.date)
    for name in ("rolling_volatility_20", "rolling_volatility_60"):
        if name not in data:
            continue
        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.plot(dates, data[name], color=colors["market"], linewidth=1.4)
        ax.set_title(f"{ticker} {name} — existing trailing feature")
        ax.set_ylabel("Daily return sample std (unannualized)")
        date_axis(ax)
        save(fig, name)

    for name in ("q_revenue_growth_yoy", "q_gross_margin"):
        if name not in data:
            continue
        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.step(dates, data[name], where="post", color=colors["quarterly"], linewidth=1.8)
        effective = parse_dates(data.q_effective_date, allow_missing=True).dropna().drop_duplicates()
        for i, day in enumerate(effective):
            if dates.iloc[0] <= day <= dates.iloc[-1]:
                ax.axvline(day, color="#888888", linestyle=":", linewidth=.8,
                           label="State effective date (sample-truncated at start)" if i == 0 else None)
        ax.set_title(f"{ticker} {name} — filing-driven quarterly states")
        ax.set_ylabel("YoY growth (fraction)" if name.endswith("growth_yoy") else "Gross margin (fraction)")
        ax.legend(fontsize=8)
        date_axis(ax)
        save(fig, name)
    return paths
