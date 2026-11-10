"""Phase A-3: OOD generalization tests on the gap_gw task (non-magnetic, 319).

Three split families, identical model protocol (TabPFN cloud, quantiles):
  random    - 5-fold KFold reference
  prototype - GroupKFold over structure prototypes (folder top dir); test
              prototypes never appear in training
  loeo      - leave-one-element-out for every element present in >= 15
              materials; test = all materials containing the element

Per split: TabPFN MAE (median), 90% CI coverage, median CI width, and
HistGB MAE for degradation comparison. The OOD-UQ claim: CI width grows
under OOD while coverage stays near nominal.

Output: data/ood_results.csv
"""

import json
from collections import Counter
from pathlib import Path

import paths
import tabpfn_backend

import numpy as np
import pandas as pd
from ase.data import chemical_symbols
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold

from run_delta_regression import DATA

MATERIALS_ROOT = paths.MATERIALS_ROOT
META_COLS = ["uid", "folder", "has_hse_json"]
TARGET_COLS = ["gap_gw", "gap_dir_gw", "vbm_gw", "cbm_gw"]


def main():
    from tabpfn_client import TabPFNRegressor

    df = pd.read_csv(DATA / "gap_gw_features.csv")
    flags = pd.read_csv(DATA / "spin_features.csv").set_index("uid")["is_magnetic"]
    df = df[df.uid.map(flags) == 0].reset_index(drop=True)
    y = df.gap_gw.to_numpy()
    feature_cols = [c for c in df.columns if c not in META_COLS + TARGET_COLS]
    X = df[feature_cols].to_numpy(dtype=float)
    X[~np.isfinite(X)] = np.nan

    elements = []
    for folder in df.folder:
        s = json.loads((MATERIALS_ROOT / folder / "structure.json").read_text())["1"]
        elements.append({chemical_symbols[z] for z in s["numbers"]})
    prototypes = np.array([f.split("/")[0] for f in df.folder])
    print(f"{len(df)} materials, {len(set(prototypes))} prototypes", flush=True)

    def evaluate(tr, te, tag, key):
        m = tabpfn_backend.make_regressor("cloud", MODEL_VERSION)()
        m.fit(X[tr], y[tr])
        q = np.asarray(m.predict(X[te], output_type="quantiles",
                                 quantiles=[0.05, 0.5, 0.95]))
        hg = HistGradientBoostingRegressor(random_state=0)
        hg.fit(X[tr], y[tr])
        row = {"split": tag, "key": key, "n_test": len(te),
               "mae_tabpfn": mean_absolute_error(y[te], q[1]),
               "mae_histgb": mean_absolute_error(y[te], hg.predict(X[te])),
               "coverage90": float(np.mean((y[te] >= q[0]) & (y[te] <= q[2]))),
               "ci_width_median": float(np.median(q[2] - q[0]))}
        print(f"{tag:10s} {key:12s} n={len(te):3d}  "
              f"tabpfn {row['mae_tabpfn']:.3f}  histgb {row['mae_histgb']:.3f}  "
              f"cov {row['coverage90']:.2f}  width {row['ci_width_median']:.3f}",
              flush=True)
        return row

    rows = []
    for fold, (tr, te) in enumerate(
            KFold(5, shuffle=True, random_state=0).split(df)):
        rows.append(evaluate(tr, te, "random", f"fold{fold}"))

    gkf = GroupKFold(n_splits=5)
    for fold, (tr, te) in enumerate(gkf.split(df, groups=prototypes)):
        rows.append(evaluate(tr, te, "prototype", f"fold{fold}"))

    counts = Counter(el for s in elements for el in s)
    loeo_elements = sorted([e for e, c in counts.items() if c >= 15],
                           key=lambda e: -counts[e])
    print(f"LOEO elements (n>=15): {loeo_elements}", flush=True)
    for el in loeo_elements:
        te = np.array([i for i, s in enumerate(elements) if el in s])
        tr = np.array([i for i, s in enumerate(elements) if el not in s])
        rows.append(evaluate(tr, te, "loeo", el))

    res = pd.DataFrame(rows)
    res.to_csv(DATA / "ood_results.csv", index=False)

    print("\n=== summary (mean over splits) ===")
    for tag, g in res.groupby("split"):
        w = np.average(g.mae_tabpfn, weights=g.n_test)
        wh = np.average(g.mae_histgb, weights=g.n_test)
        cov = np.average(g.coverage90, weights=g.n_test)
        wid = np.average(g.ci_width_median, weights=g.n_test)
        print(f"{tag:10s} tabpfn {w:.3f}  histgb {wh:.3f}  "
              f"coverage {cov:.3f}  ci_width {wid:.3f}")


if __name__ == "__main__":
    main()
