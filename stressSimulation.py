"""Reproduce the additional artificial certificate and extraction stress tests."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from mainSimulation import generate_data, estimate_standardized_effect

SETTINGS = [
    ("AI extraction error", "low extraction error", {"sigma_ai": 0.10}),
    ("AI extraction error", "moderate extraction error", {"sigma_ai": 0.60}),
    ("AI extraction error", "high extraction error", {"sigma_ai": 1.20}),
    ("Post index partial leakage", "no leakage lambda 0", {"leakage": 0.0}),
    ("Post index partial leakage", "partial leakage lambda 0.25", {"leakage": 0.25}),
    ("Post index partial leakage", "partial leakage lambda 0.50", {"leakage": 0.50}),
    ("Certificate role misclassification", "misclassification 0 percent", {"misclass": 0.0}),
    ("Certificate role misclassification", "misclassification 10 percent", {"misclass": 0.10}),
    ("Certificate role misclassification", "misclassification 20 percent", {"misclass": 0.20}),
    ("Availability error", "available at time zero", {"avail_lambda": 0.0}),
    ("Availability error", "late availability lambda 0.50", {"avail_lambda": 0.50}),
    ("Availability error", "late availability lambda 1.00", {"avail_lambda": 1.00}),
    ("Positivity stress", "ordinary overlap", {"pos_stress": 1.0}),
    ("Positivity stress", "strong positivity stress", {"pos_stress": 1.8}),
]

def run_stress(replications, n, seed_base):
    rows = []
    for setting_id, (failure_mode, setting, kwargs) in enumerate(SETTINGS):
        for rep in range(replications):
            seed = seed_base + 10000 * setting_id + rep
            df = generate_data(n=n, seed=seed, **kwargs)
            true_value, estimate, se = estimate_standardized_effect(df, ["X", "D", "O", "Zpre"])
            ci_low = estimate - 1.96 * se
            ci_high = estimate + 1.96 * se
            rows.append({
                "replication": rep + 1,
                "failureMode": failure_mode,
                "setting": setting,
                "true": true_value,
                "estimate": estimate,
                "bias": estimate - true_value,
                "se": se,
                "ciLow": ci_low,
                "ciHigh": ci_high,
                "covered": int(ci_low <= true_value <= ci_high),
            })
    return pd.DataFrame(rows)

def summarize(df):
    return df.groupby(["failureMode", "setting"], as_index=False).agg(
        bias=("bias", "mean"),
        rmse=("bias", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        coverage=("covered", "mean"),
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed-base", type=int, default=20261000)
    parser.add_argument("--outdir", default="iseSim/final")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    reps = run_stress(args.replications, args.n, args.seed_base)
    reps.to_csv(outdir / "stressReplicates.csv", index=False)
    summary = summarize(reps)
    summary.to_csv(outdir / "stressSummary.csv", index=False)
    with open(outdir / "stressSeedSchedule.json", "w", encoding="utf-8") as f:
        json.dump({"stressSimulationSeedBase": args.seed_base, "replications": args.replications, "n": args.n}, f, indent=2)
    print(summary)

if __name__ == "__main__":
    main()
