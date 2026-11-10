# Beyond the Computed: Calibrated Quasiparticle Band Gaps for 3,500+ 2D Semiconductors

Code and data release for the paper accepted at the **AI for Accelerated Materials Design (AI4Mat) workshop, NeurIPS 2026**.

**Authors:** Jiwoo Lee, Junyoung Lee, Minseong Kim, Jinwon Yoon (TendiLab)

```bibtex
@inproceedings{lee2026beyond,
  title     = {Beyond the Computed: Calibrated Quasiparticle Band Gaps for 3,500+ 2D Semiconductors},
  author    = {Lee, Jiwoo and Lee, Junyoung and Kim, Minseong and Yoon, Jinwon},
  booktitle = {NeurIPS 2026 Workshop on AI for Accelerated Materials Design (AI4Mat)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=MnKEUk0UvK}
}
```

This repository is the de-anonymized copy of the review-time mirror
(`anonymous.4open.science/r/tabpfn-gw-2d-DC13`); the contents are unchanged.

Code and derived data behind the paper. Everything here reproduces a number,
a table or a figure in the manuscript. The raw C2DB dump is not redistributed.

## Layout

```
scripts/    analysis code, in pipeline order
data/       derived datasets, cross-validation results, the public release CSVs
figures/    manuscript figures (PNG 300 dpi + PDF)
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Cloud TabPFN needs your own account:

```bash
python -c "from tabpfn_client import init; init()"
```

## Which TabPFN produced these numbers

This matters more than it usually would, so it is stated explicitly.

| | version | pinned? |
|---|---|---|
| local backend | `tabpfn==2.0.9` | **yes** - by the installed package |
| cloud backend | `tabpfn-client==0.3.1`, model `v3_default` | **yes** - pinned by `--model-version` |

The client's default is `model_path="auto"`, which defers the choice to the
server: the served model can change without anything in this repository
changing. **Every cloud call in this release therefore goes through
`scripts/tabpfn_backend.py`, which pins a named model version.**

```bash
python scripts/run_task_scaling.py --backend cloud --model-version v3   # default
python scripts/tabpfn_backend.py                                        # list versions
```

| `--model-version` | resolves to | reproducible |
|---|---|---|
| `v2` | `v2_default` | yes |
| `v2.5` | `v2.5_default` | yes |
| `v2.6` | `v2.6_default` | yes |
| **`v3`** (default) | `v3_default` | yes |
| `auto` | server's choice | **no** - kept only to reproduce older runs |

**Results committed before this module was added were produced with `auto`, so
their model version is not recorded.** `python scripts/tabpfn_backend.py
identify` re-runs the scalar-gap cross-validation under each pinned version and
compares against the committed 0.0837 eV. Measured:

| version | gap MAE (eV) | difference from the committed value |
|---|---|---|
| `v2` | 0.0922 | +0.0085 |
| `v2.5` | 0.0842 | +0.0005 |
| `v2.6` | 0.1029 | +0.0192 |
| **`v3`** | **0.0838** | **+0.0001** |

`v3` and `v2.5` both reproduce the committed number and **cannot be told apart
here**: they differ by 0.0004 eV against a fold standard deviation of 0.015 eV.
Report this as insensitivity to the version rather than as an identification.

The spread is the argument against `auto`: `v2.6` is 22% worse than `v3` on this
task, so a change in the server's default would silently move a published
number. Files carrying a `_v3` tag were produced with the version pinned.

`scripts/provenance.py` records all of this next to every result file:

```bash
python scripts/provenance.py cloud v3     # print the current stamp
```

```python
from provenance import stamp
stamp("delta_cv_results_cloud_nomag.csv", backend="cloud",
      config=dict(train_rows=10000, folds=5, seed=0))
```

Use `--backend local` for a bit-reproducible run. It is slower and, on the
scalar gap task, slightly less accurate (0.116 vs 0.113 eV), which is the price
of pinning the model.

## Reproducing

Paths resolve through `scripts/paths.py`: `<repo>/data`, `<repo>/figures`,
`<repo>/materials`, each overridable by `PBEHSE_DATA`, `PBEHSE_FIGURES`,
`PBEHSE_MATERIALS`.

**The build stage is the only part that needs the raw C2DB dump.** Obtain it
from [C2DB](https://c2db.fysik.dtu.dk) and point `PBEHSE_MATERIALS` at it. Every
experiment below runs from the shipped `data/` alone.

```
build_hse_dos.py -> build_gap_features.py -> build_delta_dataset.py -> build_screening_features.py   [needs materials/]
  run_gap_gw_regression.py --backend cloud --drop-magnetic --tag nomag
  run_delta_regression.py  --backend cloud --drop-magnetic --tag nomag      main experiment
  run_baseline_tuning.py   --drop-magnetic                                  tuned baselines, no backend needed
  run_material_curve.py    --backend cloud --drop-magnetic                  material-count curve + leakage control
  run_task_scaling.py      --backend cloud --model-version v3                task-difficulty exponents
  run_delta_uq.py / run_band_edges.py / run_screening{,_tier2}.py / run_ood_splits.py / run_active_learning.py
  run_significance.py                                                       all statistical tests
  make_figures.py / make_task_scaling_fig.py
```

## Statistical testing

`run_significance.py` reports every claim four ways and writes
`data/significance.csv`. With few cross-validation folds a paired t-test is easy
to over-read, so the weakest of the four governs:

- **paired t** - the conventional number
- **exact sign-flip permutation** - assumption-free. With $n$ folds only $2^n$
  sign assignments exist, so the smallest attainable two-sided $p$ is $2/2^n$:
  **0.0625 at $n=5$, 0.002 at $n=10$.** A t-test $p$ below that floor is produced
  by the normality assumption, not by the data.
- **Holm** - within each family of tests the paper quotes side by side
- **leave-one-fold-out jackknife** - at this $n$, more honest than any $p$

Current verdicts:

| claim | folds | verdict |
|---|---|---|
| task-scaling: TabPFN vs tuned HistGB at the largest count | 10 | **robust** (exact $p$ = 0.002, 0/10 jackknife failures) |
| task-scaling: curve still falling at the largest count | 10 | **robust** |
| tuned baseline: $1.9\times10^6$ rows beat $10^4$ | 5 | **robust** (5/5 folds, $p$ = 0.013) |
| row-level split understates the error | 5 | 67%, fold scatter 0.0006 eV |
| $\Delta\varepsilon$ curve, individual intervals | 5 | **fragile** - at or below the $n=5$ floor; four of six fail under jackknife |

The five-fold $\Delta\varepsilon$ intervals should be reported as a trend over
the whole curve rather than interval by interval, or re-run at ten folds. The
ten-fold task-scaling results carry the scaling claim.

## Data

| file | contents |
|---|---|
| `screening_release_full.csv` | **main release** - predicted gap, VBM, CBM with 90% intervals for 3,534 non-magnetic 2D semiconductors (tier 1 HSE-informed 2,613 + tier 2 PBE-only 921) |
| `band_edge_release.csv` | vacuum-aligned band edges with intervals (tier 1) |
| `photocatalyst_candidates.csv` | water-splitting candidates, **tier 1 only** (441 rows, 124 conservative). The 585 / 160 figures quoted in the paper are the tier-1 + tier-2 totals from `screening_release_full.csv`; state which you mean |
| `delta_eps_dataset.npz` | 2,501,600 states from 261 materials, target $\Delta\varepsilon$ |
| `gap_gw_features.csv` | material-level features, 339 $G_0W_0$ materials |
| `material_curve.csv`, `baseline_tuning.csv`, `task_scaling.csv` | the scaling and baseline experiments |
| `significance.csv` | output of `run_significance.py` |
| `delta_uq_calibration.npz` | quantile levels, predicted quantiles and targets for the calibration analysis |
| `literature_gw.csv` | 13 published $G_0W_0$ values with citations; note which include spin-orbit coupling |

## Conventions that silently corrupt results if broken

- **Splits are over materials, never rows.** 2.5M rows come from 261 materials;
  a row-level split understates the error by 67% (measured, not asserted).
- **All energies are vacuum-referenced.** Raw `hse.json` and
  `results-asr.gw.json` eigenvalues are on the calculator scale; `data.json`
  scalars are already vacuum-referenced. Subtract `evac` before comparing.
- **No $GW$-derived feature.** `gap_gw`, `gap_dir_gw`, `vbm_gw`, `cbm_gw` are
  targets. `screening_features_all.csv` carries `gap_gw`, so filter with the
  `TARGET_COLS` / `META_COLS` constants rather than by column-name pattern;
  `run_task_scaling.py` has an explicit guard because this leak was made once.
- `emass_*` is `inf` for flat bands - every consumer maps non-finite to `NaN`.
- Never use a C2DB `uid` as a filename: `2BiIn` and `2BiIN` collide on
  case-insensitive filesystems.
- Scope is **non-magnetic**. The 20 magnetic materials carry about five times
  the error and spin descriptors do not rescue them.

## Not included

- The raw C2DB dump (see above).
- `extract_mace_embeddings.py` and the DOS-showcase script: the first reproduces
  a negative result already summarised in the paper, the second depends on a
  metric whose generating code is not in the repository.
- The prospective $G_0W_0$ validation bundle: that calculation has not been run,
  and the paper says so.

## License and attribution

Code: MIT (see `LICENSE`). Derived data build on
[C2DB](https://c2db.fysik.dtu.dk) (Haastrup et al., *2D Mater.* **5**, 042002
(2018); Gjerding et al., *2D Mater.* **8**, 044002 (2021)) - cite C2DB when
using them. TabPFN: Hollmann et al., *Nature* **637**, 319-326 (2025).
