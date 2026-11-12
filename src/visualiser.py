"""
src/visualiser.py
Post-run visualisation and occupancy analytics dashboard.

Generates:
  • Occupancy timeline chart per seat
  • Heatmap of seat utilisation
  • Daily utilisation summary
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger


def load_log(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def plot_occupancy_timeline(
    csv_path: str | Path,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot occupied/vacant state per seat over time."""
    try:
        import matplotlib.dates as mdates
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping timeline plot.")
        return

    df = load_log(csv_path)
    if df.empty:
        logger.warning("No data to plot.")
        return

    seats = sorted(df["seat_id"].unique())
    n = len(seats)

    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = {"occupied": "#2196F3", "vacant": "#E0E0E0", "unknown": "#9E9E9E"}

    for ax, seat_id in zip(axes, seats):
        seat_df = df[df["seat_id"] == seat_id].sort_values("timestamp")

        for i in range(len(seat_df) - 1):
            row = seat_df.iloc[i]
            next_row = seat_df.iloc[i + 1]
            color = colors.get(row["state"], "#9E9E9E")
            ax.barh(
                0,
                width=(next_row["timestamp"] - row["timestamp"]).total_seconds() / 3600,
                left=mdates.date2num(row["timestamp"]),
                height=0.8,
                color=color,
            )

        ax.set_ylabel(seat_id, rotation=0, ha="right", va="center", fontsize=9)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlabel("Time")
    fig.suptitle("Seat Occupancy Timeline", fontsize=14, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color=colors["occupied"], label="Occupied"),
        mpatches.Patch(color=colors["vacant"],   label="Vacant"),
    ]
    fig.legend(handles=legend_patches, loc="upper right")
    plt.tight_layout()

    if output_path:
        plt.savefig(str(output_path), dpi=150)
        logger.info(f"[Visualiser] Timeline saved → {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_utilisation_heatmap(
    csv_path: str | Path,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Heatmap: seats × hours of day, colour = occupancy rate."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not installed — skipping heatmap.")
        return

    df = load_log(csv_path)
    if df.empty:
        logger.warning("No data for heatmap.")
        return

    df["hour"] = df["timestamp"].dt.hour
    df["occupied_bin"] = (df["state"] == "occupied").astype(int)

    pivot = df.groupby(["seat_id", "hour"])["occupied_bin"].mean().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.6 + 2)))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        vmin=0, vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "Occupancy Rate"},
    )
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Seat")
    ax.set_title("Seat Utilisation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        plt.savefig(str(output_path), dpi=150)
        logger.info(f"[Visualiser] Heatmap saved → {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def print_utilisation_table(csv_path: str | Path) -> pd.DataFrame:
    """Print and return a utilisation summary table."""
    df = load_log(csv_path)
    if df.empty:
        print("No data available.")
        return pd.DataFrame()

    df = df.sort_values(["seat_id", "timestamp"])
    rows = []

    for seat_id, group in df.groupby("seat_id"):
        group = group.reset_index(drop=True)
        total_occupied_sec = 0.0
        total_sec = 0.0

        for i in range(len(group) - 1):
            duration = (
                group.loc[i + 1, "timestamp"] - group.loc[i, "timestamp"]
            ).total_seconds()
            total_sec += duration
            if group.loc[i, "state"] == "occupied":
                total_occupied_sec += duration

        util = total_occupied_sec / total_sec if total_sec > 0 else 0.0
        rows.append({
            "Seat": seat_id,
            "Total Time (min)":    round(total_sec / 60, 1),
            "Occupied (min)":      round(total_occupied_sec / 60, 1),
            "Utilisation (%)":     round(util * 100, 1),
        })

    summary_df = pd.DataFrame(rows).sort_values("Utilisation (%)", ascending=False)

    print("\n" + "═" * 55)
    print("  SEAT UTILISATION SUMMARY")
    print("═" * 55)
    print(summary_df.to_string(index=False))
    print("═" * 55 + "\n")

    return summary_df


def generate_all(
    csv_path: str | Path,
    output_dir: str | Path = "output/",
    show: bool = False,
) -> None:
    """Generate all visualisations from a log CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_occupancy_timeline(
        csv_path,
        output_path=out / "timeline.png",
        show=show,
    )
    plot_utilisation_heatmap(
        csv_path,
        output_path=out / "heatmap.png",
        show=show,
    )
    summary = print_utilisation_table(csv_path)
    summary.to_csv(out / "utilisation_summary.csv", index=False)
    logger.info(f"[Visualiser] All outputs saved → {out}")
