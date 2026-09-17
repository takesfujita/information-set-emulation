"""Create summary CSV and LaTeX tables from replicate-level simulation outputs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

OUTDIR = Path("iseSim/final")
TABLEDIR = Path("../tables")

ORDER = [
    "Structured EHR estimator",
    "Naive AI estimator",
    "ISE causal estimator",
    "Design erasure estimator",
    "Post index leakage estimator",
    "Oracle prospective estimator",
]

LATEX_END = r"\\"

def fmt(x):
    return f"{x:.3f}"

def main_summary():
    reps = pd.read_csv(OUTDIR / "mainReplicates.csv")
    summary = reps.groupby("method", as_index=False).agg(
        true=("true", "mean"),
        estimate=("estimate", "mean"),
        bias=("bias", "mean"),
        rmse=("bias", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        meanSE=("se", "mean"),
        coverage=("covered", "mean"),
    )
    summary["method"] = pd.Categorical(summary["method"], ORDER, ordered=True)
    summary = summary.sort_values("method")
    summary.to_csv(OUTDIR / "mainSummary.csv", index=False)
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        f"Method & True & Estimate & Bias & RMSE & Mean SE & Coverage {LATEX_END}",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(f"{row['method']} & {fmt(row.true)} & {fmt(row.estimate)} & {fmt(row.bias)} & {fmt(row.rmse)} & {fmt(row.meanSE)} & {fmt(row.coverage)} {LATEX_END}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLEDIR / "TableMainSimulation.tex").write_text("\n".join(lines), encoding="utf-8")
    return summary

def frontier():
    df = pd.read_csv(OUTDIR / "frontierSummary.csv")
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        fr"Evidence & Retained worlds & $r_K(p)$ & Center RMSE & RMSE bound & Worst-world RMSE & Meets {LATEX_END}",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        state = str(row.evidenceState)
        state = state.replace("AI plus source/availability audit Ktheta+E", r"Typed + audit $K_\theta\!\vee\!E$")
        state = state.replace("Post-index source certificate Kpost", r"Post-index $K_{post}$")
        state = state.replace("AI typed lift Ktheta", r"Typed $K_\theta$")
        state = state.replace("Flat table K0", r"Flat $K_0$")
        worlds = str(row.worlds).replace(",", ", ")
        lines.append(f"{state} & {worlds} & {fmt(row.radius)} & {fmt(row.centerRMSE)} & {fmt(row.frontierRMSE)} & {fmt(row.worstWorldRMSE)} & {row.meetsFrontier} {LATEX_END}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLEDIR / "TableFrontier.tex").write_text("\n".join(lines), encoding="utf-8")
    return df

def stress_summary():
    reps = pd.read_csv(OUTDIR / "stressReplicates.csv")
    summary = reps.groupby(["failureMode", "setting"], as_index=False).agg(
        bias=("bias", "mean"),
        rmse=("bias", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        coverage=("covered", "mean"),
    )
    order = reps[["failureMode", "setting"]].drop_duplicates().reset_index(drop=True)
    order["rowOrder"] = range(len(order))
    summary = summary.merge(order, on=["failureMode", "setting"], how="left").sort_values("rowOrder").drop(columns=["rowOrder"])
    summary.to_csv(OUTDIR / "stressSummary.csv", index=False)
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        f"Failure mode & Setting & Bias & RMSE & Coverage {LATEX_END}",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(f"{row.failureMode} & {row.setting} & {fmt(row.bias)} & {fmt(row.rmse)} & {fmt(row.coverage)} {LATEX_END}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLEDIR / "TableAdditionalStress.tex").write_text("\n".join(lines), encoding="utf-8")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate summary CSV files and LaTeX tables from replicate-level outputs.")
    parser.add_argument("--outdir", default="iseSim/final", help="Directory containing mainReplicates.csv and stressReplicates.csv.")
    parser.add_argument("--tabledir", default="../tables", help="Directory for LaTeX table outputs.")
    args = parser.parse_args()
    OUTDIR = Path(args.outdir)
    TABLEDIR = Path(args.tabledir)
    TABLEDIR.mkdir(parents=True, exist_ok=True)
    sm = main_summary()
    frontier()
    stress_summary()
