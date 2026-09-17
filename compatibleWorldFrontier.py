"""Exact compatible-world calibration for the certificate frontier.

The script fixes one strictly positive observed law P(A,Z,Y) and constructs
three causal worlds that induce that same law but assign different causal roles
to Z.  The information radius is therefore computed from an actual fiber, not
from the bias of an estimator.  Monte Carlo sampling is used only to estimate
the sampling error of the plug-in Chebyshev center.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


WORLD_LABELS = {
    "B": "pre-treatment state",
    "M": "post-treatment mediator",
    "Y": "outcome-adjacent proxy",
}

STATE_WORLDS = {
    "Flat table K0": ["B", "M", "Y"],
    "AI typed lift Ktheta": ["B", "M"],
    "AI plus source/availability audit Ktheta+E": ["B"],
    "Post-index source certificate Kpost": ["M", "Y"],
}


def observed_law():
    """Return the 2x2x2 array p[a,z,y] for the common observed law."""
    p_z = np.array([0.5, 0.5])
    p_a1_z = np.array([0.2, 0.8])
    p_y1_az = np.array([[0.10, 0.40], [0.30, 0.70]])
    p = np.zeros((2, 2, 2), dtype=float)
    for z in (0, 1):
        for a in (0, 1):
            pa = p_a1_z[z] if a == 1 else 1.0 - p_a1_z[z]
            py = p_y1_az[a, z]
            p[a, z, 1] = p_z[z] * pa * py
            p[a, z, 0] = p_z[z] * pa * (1.0 - py)
    if not np.isclose(p.sum(), 1.0) or np.any(p <= 0):
        raise ValueError("The common observed law must be strictly positive and sum to one.")
    return p


def role_functionals(p, missing_y1_untreated=0.10, missing_y0_treated=0.90):
    """Compute the three world-specific targets from a common observed law.

    B uses the standardized contrast over P(Z).  M treats A as exogenous and Z
    as post-treatment, giving the crude interventional contrast.  Y treats Z as
    outcome-adjacent and allows latent treatment selection; its two unobserved
    counterfactual means are fixed sensitivity coordinates in [0,1].
    """
    p_z = p.sum(axis=(0, 2))
    p_a = p.sum(axis=(1, 2))
    q_az = np.zeros((2, 2), dtype=float)
    for a in (0, 1):
        for z in (0, 1):
            denom = p[a, z, :].sum()
            q_az[a, z] = p[a, z, 1] / denom
    q_a = np.array([p[a, :, 1].sum() / p_a[a] for a in (0, 1)])

    psi_b = float(np.sum(p_z * (q_az[1] - q_az[0])))
    psi_m = float(q_a[1] - q_a[0])
    pi = float(p_a[1])
    psi_y = float(
        pi * q_a[1]
        + (1.0 - pi) * missing_y1_untreated
        - pi * missing_y0_treated
        - (1.0 - pi) * q_a[0]
    )
    return {"B": psi_b, "M": psi_m, "Y": psi_y}


def estimate_from_counts(counts):
    return role_functionals(counts / counts.sum())


def center(values):
    return 0.5 * (min(values) + max(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="iseSim/final")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--replications", type=int, default=5000)
    parser.add_argument("--ehr-n", type=int, default=1000)
    parser.add_argument("--prospective-n", type=int, default=300)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    p = observed_law()
    targets = role_functionals(p)
    cell_prob = p.reshape(-1)

    state_estimates = {state: [] for state in STATE_WORLDS}
    prospective_estimates = []
    replicate_rows = []
    for rep in range(args.replications):
        ehr_counts = rng.multinomial(args.ehr_n, cell_prob).reshape(2, 2, 2)
        role_est = estimate_from_counts(ehr_counts)
        row = {"replication": rep + 1}
        for state, worlds in STATE_WORLDS.items():
            c_hat = center([role_est[w] for w in worlds])
            state_estimates[state].append(c_hat)
            row[state] = c_hat
        prospective_counts = rng.multinomial(args.prospective_n, cell_prob).reshape(2, 2, 2)
        prospective_hat = estimate_from_counts(prospective_counts)["B"]
        prospective_estimates.append(prospective_hat)
        row["Prospective reference KP"] = prospective_hat
        replicate_rows.append(row)

    prospective_estimates = np.asarray(prospective_estimates)
    prospective_ref_rmse = float(np.sqrt(np.mean((prospective_estimates - targets["B"]) ** 2)))

    summary_rows = []
    for state, worlds in STATE_WORLDS.items():
        values = np.asarray([targets[w] for w in worlds], dtype=float)
        c = center(values)
        r = 0.5 * (float(values.max()) - float(values.min()))
        estimates = np.asarray(state_estimates[state], dtype=float)
        center_bias = float(np.mean(estimates - c))
        center_rmse = float(np.sqrt(np.mean((estimates - c) ** 2)))
        worst_world_rmse = max(
            float(np.sqrt(np.mean((estimates - targets[w]) ** 2))) for w in worlds
        )
        frontier_rmse = r + center_rmse
        summary_rows.append(
            {
                "evidenceState": state,
                "worlds": ",".join(worlds),
                "lower": float(values.min()),
                "upper": float(values.max()),
                "center": c,
                "radius": r,
                "centerBias": center_bias,
                "centerRMSE": center_rmse,
                "frontierRMSE": frontier_rmse,
                "worstWorldRMSE": worst_world_rmse,
                "prospectiveRefRMSE": prospective_ref_rmse,
                "meetsFrontier": "Yes" if frontier_rmse <= prospective_ref_rmse else "No",
            }
        )

    world_rows = [
        {
            "world": symbol,
            "roleInterpretation": WORLD_LABELS[symbol],
            "estimandValue": targets[symbol],
        }
        for symbol in ("B", "M", "Y")
    ]
    pd.DataFrame(world_rows).to_csv(outdir / "frontierWorlds.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(outdir / "frontierSummary.csv", index=False)
    pd.DataFrame(replicate_rows).to_csv(outdir / "frontierReplicates.csv", index=False)
    with (outdir / "frontierSeedSchedule.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "seed": args.seed,
                "replications": args.replications,
                "ehrSampleSize": args.ehr_n,
                "prospectiveSampleSize": args.prospective_n,
                "missingY1Untreated": 0.10,
                "missingY0Treated": 0.90,
            },
            stream,
            indent=2,
        )

    print(pd.DataFrame(world_rows).round(4).to_string(index=False))
    print(pd.DataFrame(summary_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
