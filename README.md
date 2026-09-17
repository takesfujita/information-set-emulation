# Information Set Emulation

## Causal Certificates for AI Derived EHR Features

**Takes Fujita** — VRI  
**Nobutaka Hattori** — Department of Neurology, Juntendo University School of Medicine

**Paper:** [arXiv:2609.17777](https://arxiv.org/abs/2609.17777)

This repository contains the manuscript source, synthetic experiments, reporting calculators, and certificate template for *Information Set Emulation: Causal Certificates for AI Derived EHR Features*.

## Overview

An electronic health record may contain useful information that was recorded after treatment began or after an outcome occurred. Extracting that information accurately does not, by itself, make it suitable for causal adjustment.

Information set emulation asks whether information reconstructed from retrospective records can support the eligibility, treatment, observation, and comparison conditions of a fixed target trial. A typed ledger records each feature's source, clinical and recording times, decision-time availability, proposed causal roles, and unresolved ambiguity. Causal certificates organize the evidence supporting those roles.

The paper connects this evidence to partial identification and decision theory. Compatible causal worlds define a set of possible values for one locked scalar estimand. For a nonempty compact compatible image, the classical squared Chebyshev-radius identity characterizes residual minimax mean squared error on the fixed observed law. The contribution is its integration with the EHR observation model and certificate framework. Sampling uncertainty is addressed separately.

All experiments and note-like records in this repository are synthetic; no real patient records are included.

## Repository contents

| Path | Contents |
| --- | --- |
| [manuscript.tex](manuscript.tex) | Main manuscript, including the bibliography |
| [tables/](tables/) | Tables included by the manuscript |
| [iseSim/final/frontierPlot.png](iseSim/final/frontierPlot.png) | Compatible-world frontier figure |
| [anc/iseSim/final/](anc/iseSim/final/) | Main, stress, frontier, and compression-drift results |
| [anc/phase0SyntheticOutputs/](anc/phase0SyntheticOutputs/) | Synthetic notes, audit diagnostics, and analysis-spread examples |
| [anc/fullSeedSchedule.json](anc/fullSeedSchedule.json) | Simulation seeds and settings |
| [anc/certificateTemplate.json](anc/certificateTemplate.json) | Blank causal-certificate template |
| [anc/CompatibleReportingCalculator.py](anc/CompatibleReportingCalculator.py) | Compatible reporting with disconnected interval unions |
| [anc/ValueOfTypingCalculator.py](anc/ValueOfTypingCalculator.py) | Reporting-score and minimax calculations |
| [anc/verifyReproduction.py](anc/verifyReproduction.py) | Reproduction workflow and output comparisons |

## Install dependencies and check integrity

Run these commands from the repository root, preferably in a virtual environment:

```bash
python3 -m pip install -r anc/requirements.txt
python3 anc/verifyManifest.py
```

The integrity check verifies the package manifest and the sizes and SHA-256 hashes of the listed files.

The reference environment is Python 3.12.13, NumPy 2.3.5, pandas 2.2.3, Matplotlib 3.10.8, and Pillow 12.3.0. The requirements file lists dependencies without pinning their versions. To match the reference package versions, use:

```bash
python3 -m pip install numpy==2.3.5 pandas==2.2.3 matplotlib==3.10.8 pillow==12.3.0
```

Matching the environment is particularly relevant to exact figure-pixel comparisons.

## Reproduce the experiments

From the repository root:

```bash
python3 anc/verifyReproduction.py --run-dir ../ise_reproduction
```

The output directory must not already exist. The script runs:

- Main simulation: 1,000 replications.
- Stress experiments: 200 replications per setting.
- Common-law finite-world frontier: 5,000 replications.
- Compression drift: exact enumeration of a 16-state example.
- Synthetic Phase 0 diagnostics and the analysis-spread illustration.
- Table and figure generation.

It compares 22 outputs with the bundled references: CSV numeric values with relative and absolute tolerances of `1e-12`, JSON by content, LaTeX tables by exact bytes, and the frontier figure by exact rendered pixels. Results are written to the separate output directory.

## Compile the manuscript

From the repository root, using a LaTeX installation with the required packages:

```bash
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
```

Keep `tables/` and `iseSim/` alongside `manuscript.tex`. The bibliography is embedded, so BibTeX and Biber are not required. Compilation does not require shell escape.

## Interpretation

The exact finite-world frontier concerns compatible worlds sharing one observed law and one locked target. The synthetic Phase 0 example has a different role: Table 15 reports a role-only admission diagnostic, and Table 16 reports prespecified role-specific analysis spread. That spread is not an exact compatible-fiber radius, and its retained analysis lists are not inferred from the audit error rates.

The certificate template is unfilled. Null and empty fields do not establish certification. The reporting calculators implement the specified calculations; the source, availability, and validation conditions must be established for the study being analyzed.

## Citation

If you use this work, code, or synthetic data, please cite the paper:

```bibtex
@misc{fujita2026informationsetemulation,
  author        = {Fujita, Takes and Hattori, Nobutaka},
  title         = {{Information Set Emulation: Causal Certificates for AI Derived EHR Features}},
  year          = {2026},
  eprint        = {2609.17777},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2609.17777}
}
```

Package citation information is also available in [anc/CITATION.cff](anc/CITATION.cff).

## License

The supplementary software and its accompanying software documentation are provided under the [MIT License](anc/LICENSE).

The manuscript was submitted under the [arXiv.org perpetual, non-exclusive distribution license](https://arxiv.org/licenses/nonexclusive-distrib/1.0/). The software license does not apply to the manuscript.
