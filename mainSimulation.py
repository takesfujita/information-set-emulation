"""Reproduce the main artificial EHR simulation and frontier table.

This is a fully synthetic simulation. It does not use patient data.
It generates an artificial prospective information set W=(X,D,J,O), an EHR
frame indicator, treatment choice, outcome observation, and several AI-derived
features. Results are written at replicate level so that tables can be
recomputed by a reviewer.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.linalg import pinv

METHODS = {
    "Structured EHR estimator": ["X", "D", "O"],
    "Naive AI estimator": ["X", "D", "O", "Zpre", "Zpost", "Zcomp"],
    "ISE causal estimator": ["X", "D", "O", "Zpre"],
    "Design erasure estimator": ["X", "O", "Zcomp"],
    "Post index leakage estimator": ["X", "D", "O", "Zpost"],
    "Oracle prospective estimator": ["X", "D", "J", "O"],
}

def expit(x):
    return 1.0 / (1.0 + np.exp(-x))

def generate_data(n, seed, sigma_ai=0.10, leakage=0.0, misclass=0.0, avail_lambda=0.0, pos_stress=1.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=n)
    J = 0.65 * X + rng.normal(size=n)
    O = 0.35 * X + 0.55 * J + rng.normal(size=n)
    D = rng.binomial(1, expit(-0.15 + 0.55 * X - 0.25 * O + 0.25 * J))
    S = rng.binomial(1, expit(1.20 + 0.35 * X + 0.55 * D + 0.45 * O))
    A = rng.binomial(1, expit(pos_stress * (-0.25 + 0.60 * X + 0.85 * D + 1.05 * J + 0.60 * O)))
    eps_y = rng.normal(size=n)
    Y0 = 0.30 + 0.75 * X + 0.80 * D + 1.00 * J + 0.65 * O + eps_y
    treatment_effect = 0.60 + 0.30 * J
    Y = Y0 + A * treatment_effect
    Delta = rng.binomial(1, expit(1.05 + 0.25 * A + 0.45 * X + 0.50 * J + 0.75 * O + 0.35 * D))
    M = 0.65 * A + 0.55 * J + 0.25 * X + 0.20 * D + rng.normal(size=n)
    Zpre = J + rng.normal(scale=sigma_ai, size=n)
    Zcomp = 0.60 * X + 0.85 * J + rng.normal(scale=0.45, size=n)
    Zpost = 0.90 * M + 0.80 * Y + rng.normal(scale=0.55, size=n)
    if leakage:
        Zpre = (Zpre + leakage * Zpost) / np.sqrt(1.0 + leakage ** 2)
    if misclass:
        Zpre = (1.0 - misclass) * Zpre + misclass * Zpost
    if avail_lambda:
        late_source = 0.80 * A + 0.80 * Y + 0.50 * M
        Zpre = (Zpre + avail_lambda * late_source) / np.sqrt(1.0 + avail_lambda ** 2)
    return pd.DataFrame({
        "X": X, "J": J, "O": O, "D": D, "S": S, "A": A, "Delta": Delta,
        "Y": Y, "treatmentEffect": treatment_effect, "Zpre": Zpre, "Zcomp": Zcomp, "Zpost": Zpost,
    })

def design_matrix(df, covariates):
    columns = ["intercept", "A"] + list(covariates) + [f"A:{c}" for c in covariates]
    Xmat = np.ones((len(df), len(columns)))
    Xmat[:, 1] = df["A"].to_numpy()
    for k, c in enumerate(covariates, start=2):
        Xmat[:, k] = df[c].to_numpy()
    for k, c in enumerate(covariates, start=2 + len(covariates)):
        Xmat[:, k] = df["A"].to_numpy() * df[c].to_numpy()
    return Xmat, columns

def estimate_standardized_effect(df, covariates):
    target = df["S"] == 1
    observed = (df["S"] == 1) & (df["Delta"] == 1)
    dfo = df.loc[observed].copy()
    Xmat, columns = design_matrix(dfo, covariates)
    y = dfo["Y"].to_numpy()
    inv = pinv(Xmat.T @ Xmat)
    beta = inv @ Xmat.T @ y
    resid = y - Xmat @ beta
    n_obs = len(y)
    p = len(columns)
    meat = (Xmat * resid[:, None]).T @ (Xmat * resid[:, None])
    cov_beta = (n_obs / max(1, n_obs - p)) * inv @ meat @ inv
    target_df = df.loc[target]
    g = np.zeros(len(columns))
    g[1] = 1.0
    for idx, c in enumerate(covariates, start=2 + len(covariates)):
        g[idx] = target_df[c].mean()
    estimate = float(g @ beta)
    se = float(np.sqrt(max(0.0, g @ cov_beta @ g)))
    true_value = float(target_df["treatmentEffect"].mean())
    return true_value, estimate, se

def run_main(replications, n, seed_base):
    rows = []
    for rep in range(replications):
        df = generate_data(n=n, seed=seed_base + rep)
        for method, covariates in METHODS.items():
            true_value, estimate, se = estimate_standardized_effect(df, covariates)
            ci_low = estimate - 1.96 * se
            ci_high = estimate + 1.96 * se
            rows.append({
                "replication": rep + 1,
                "scenario": "main",
                "method": method,
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
    return df.groupby(["scenario", "method"], as_index=False).agg(
        true=("true", "mean"),
        estimate=("estimate", "mean"),
        bias=("bias", "mean"),
        rmse=("bias", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        meanSE=("se", "mean"),
        coverage=("covered", "mean"),
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=1000)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed-base", type=int, default=20260000)
    parser.add_argument("--outdir", default="iseSim/final")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    reps = run_main(args.replications, args.n, args.seed_base)
    reps.to_csv(outdir / "mainReplicates.csv", index=False)
    summary = summarize(reps)
    summary.to_csv(outdir / "mainSummary.csv", index=False)
    with open(outdir / "mainSeedSchedule.json", "w", encoding="utf-8") as f:
        json.dump({"mainSimulationSeedBase": args.seed_base, "replications": args.replications, "n": args.n}, f, indent=2)
    print(summary)

if __name__ == "__main__":
    main()
