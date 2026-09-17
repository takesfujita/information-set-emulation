"""Run the synthetic simulations in a clean verification directory and compare outputs.

This script verifies reproducibility of the artificial simulation outputs bundled
with the package. It does not use patient data.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageChops

COMPARE_FILES = [
    "iseSim/final/mainReplicates.csv",
    "iseSim/final/mainSummary.csv",
    "iseSim/final/stressReplicates.csv",
    "iseSim/final/stressSummary.csv",
    "iseSim/final/frontierReplicates.csv",
    "iseSim/final/frontierSummary.csv",
    "iseSim/final/frontierWorlds.csv",
    "iseSim/final/frontierSeedSchedule.json",
    "iseSim/final/compressionDriftSummary.csv",
    "iseSim/final/compressionDriftCells.csv",
    "iseSim/final/compressionDriftSupport.csv",
    "iseSim/final/compressionDriftSpecification.json",
    "tables/TableCompressionDrift.tex",
    "tables/TableMainSimulation.tex",
    "tables/TableFrontier.tex",
    "tables/TableAdditionalStress.tex",
    "phase0SyntheticOutputs/phase0FeatureLevelResults.csv",
    "phase0SyntheticOutputs/phase0ValidationMetrics.csv",
    "phase0SyntheticOutputs/phase0AnalysisSpread.csv",
    "phase0SyntheticOutputs/roleSpecificAnalysisValues.csv",
    "phase0SyntheticOutputs/syntheticNoteVignettes.csv",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_core(refp: Path, newp: Path):
    if not refp.exists() or not newp.exists():
        return False, "missing"
    if refp.suffix == ".csv":
        left = pd.read_csv(refp)
        right = pd.read_csv(newp)
        if list(left.columns) != list(right.columns) or left.shape != right.shape:
            return False, "csv schema"
        for column in left.columns:
            if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(right[column]):
                if not np.allclose(left[column], right[column], rtol=1e-12, atol=1e-12, equal_nan=True):
                    return False, f"csv numeric column {column}"
            else:
                if not left[column].fillna("<NA>").astype(str).equals(right[column].fillna("<NA>").astype(str)):
                    return False, f"csv text column {column}"
        return True, "csv numeric tolerance 1e-12"
    if refp.suffix == ".json":
        return json.loads(refp.read_text(encoding="utf-8")) == json.loads(newp.read_text(encoding="utf-8")), "json semantic"
    return sha(refp) == sha(newp), "exact bytes"


def compare_png_pixels(refp: Path, newp: Path):
    if not refp.exists() or not newp.exists() or refp.stat().st_size == 0 or newp.stat().st_size == 0:
        return False
    with Image.open(refp) as left, Image.open(newp) as right:
        if left.size != right.size or left.mode != right.mode:
            return False
        return ImageChops.difference(left, right).getbbox() is None


def run(cmd):
    print("RUN", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(Path(__file__).resolve().parent.parent / "verificationRun"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    rundir = Path(args.run_dir).resolve()
    rundir.mkdir(parents=True, exist_ok=False)
    (rundir / "iseSim" / "final").mkdir(parents=True, exist_ok=True)
    (rundir / "tables").mkdir(parents=True, exist_ok=True)

    py = sys.executable
    run([py, "mainSimulation.py", "--replications", "1000", "--n", "1000", "--seed-base", "20260000", "--outdir", str(rundir / "iseSim" / "final")])
    run([py, "stressSimulation.py", "--replications", "200", "--n", "1000", "--seed-base", "20261000", "--outdir", str(rundir / "iseSim" / "final")])
    run([py, "compatibleWorldFrontier.py", "--replications", "5000", "--ehr-n", "1000", "--prospective-n", "300", "--seed", "20260807", "--outdir", str(rundir / "iseSim" / "final")])
    run([py, "compressionDriftValidation.py", "--outdir", str(rundir / "iseSim" / "final"), "--tabledir", str(rundir / "tables")])
    run([py, "makeTables.py", "--outdir", str(rundir / "iseSim" / "final"), "--tabledir", str(rundir / "tables")])
    run([py, "makeFrontierFigure.py", "--outdir", str(rundir / "iseSim" / "final"), "--figdir", str(rundir / "iseSim" / "final")])
    run([py, "syntheticPhase0Validation.py", "--outdir", str(rundir / "phase0SyntheticOutputs"), "--seed", "20260609"])

    rows = []
    all_ok = True
    # compare selected deterministic text/csv outputs. Figure png metadata can vary by matplotlib version,
    # so figure verification is reported separately if exact hash matches.
    for ref in COMPARE_FILES:
        refp = (root.parent if ref.startswith("tables/") else root) / ref
        newp = rundir / ref
        ok, comparison = compare_core(refp, newp)
        rows.append({"file": ref, "comparison": comparison, "referenceSha256": sha(refp) if refp.exists() else None, "rerunSha256": sha(newp) if newp.exists() else None, "match": ok})
        all_ok = all_ok and ok
    # Compare rendered pixels rather than PNG metadata.
    for ref in ["iseSim/final/frontierPlot.png"]:
        refp = root.parent / ref
        newp = rundir / ref
        ok = compare_png_pixels(refp, newp)
        rows.append({"file": ref, "comparison": "exact rendered pixels", "referenceSha256": sha(refp) if refp.exists() else None, "rerunSha256": sha(newp) if newp.exists() else None, "match": ok})
        all_ok = all_ok and ok
    (rundir / "verificationReport.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("ALL CORE OUTPUTS MATCH:", all_ok)
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
