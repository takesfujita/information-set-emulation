"""Recreate the exact compatible-world certificate frontier figure."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = Path("iseSim/final")
FIGDIR = Path("iseSim/final")

def main():
    df = pd.read_csv(OUTDIR / "frontierSummary.csv")
    label_map = {
        "Flat table K0": "Flat table",
        "AI typed lift Ktheta": "Typed lift",
        "AI plus source/availability audit Ktheta+E": "Typed + audit",
        "Post-index source certificate Kpost": "Post-index",
    }
    labels = [label_map.get(x, x) for x in df["evidenceState"]]
    x = range(len(df))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x, df["radius"], label="Exact information radius")
    ax.bar(x, df["centerRMSE"], bottom=df["radius"], label="Center estimation RMSE")
    ax.scatter(x, df["worstWorldRMSE"], marker="o", color="black", label="Observed worst-world RMSE")
    ax.axhline(df["prospectiveRefRMSE"].iloc[0], linestyle="--", label="Prospective reference RMSE")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("RMSE scale")
    ax.set_title("Exact compatible-world certificate frontier")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "frontierPlot.png", dpi=220)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recreate the certificate conditional approximation frontier figure.")
    parser.add_argument("--outdir", default="iseSim/final", help="Directory containing frontierSummary.csv.")
    parser.add_argument("--figdir", default="../iseSim/final", help="Directory for figure output at the package root.")
    args = parser.parse_args()
    OUTDIR = Path(args.outdir)
    FIGDIR = Path(args.figdir) if args.figdir is not None else OUTDIR
    FIGDIR.mkdir(parents=True, exist_ok=True)
    main()
