import tabpfn_backend
"""Does the scaling exponent depend on how hard the correction is?

The Delta-eps curve says materials, not rows, are the binding constraint - but it
runs out of data at 216 materials, so it cannot say where (or whether) the curve
flattens. Two correction tasks defined on the SAME materials with the SAME
PBE-level features let us ask a sharper question:

    PBE -> HSE   2,968 non-magnetic labels    cheap correction
    PBE -> GW      319 non-magnetic labels    expensive correction

Same protocol, same features, only the target changes. If the log-log exponent
differs, the amount of data a correction needs is set by the physics it has to
recover, not by the estimator. The HSE task also extends the x-axis 12-fold,
which is the only way to see whether these curves flatten at all.

Both tasks are material-level (one row per material), so rows and materials
coincide here. That contrast with the state-level Delta-eps curve is the point.

Writes data/task_scaling.csv incrementally.

    python scripts/run_task_scaling.py --backend cloud
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_delta_regression as R  # noqa: E402  for DATA only

DATA = R.DATA
HSE_COLS = ["gap_hse", "gap_dir_hse", "vbm_hse", "cbm_hse", "efermi_hse"]
# screening_features_all.csv carries gap_gw and has_gw as well. Reuse the
# repo's own TARGET_COLS/META_COLS rather than pattern-matching column names -
# that is exactly the mistake those constants exist to prevent.
ADMIN = ["tier", "magnetic", "has_gw"]
DROP = sorted(set(HSE_COLS + R.TARGET_COLS + R.META_COLS + ADMIN))
GRIDS = {                       # training-material counts per task
    "gap_hse": [20, 40, 80, 160, 320, 640, 1280, 2671],
    "gap_gw":  [20, 40, 80, 160, 287],
}


def build_tasks():
    """Same feature matrix for both targets, so the exponents are comparable."""
    f = pd.read_csv(DATA / "screening_features_all.csv")
    feat = [c for c in f.columns if c not in DROP]
    banned = [c for c in feat if c in R.TARGET_COLS or c in HSE_COLS or "gw" in c.lower()]
    if banned:
        raise SystemExit(f"leak guard: {banned} must not be features")
    nonmag = f.magnetic == 0

    tasks = {}
    ok = nonmag & np.isfinite(pd.to_numeric(f.gap_hse, errors="coerce"))
    tasks["gap_hse"] = (f.loc[ok, feat].apply(pd.to_numeric, errors="coerce").to_numpy(float),
                        pd.to_numeric(f.loc[ok, "gap_hse"]).to_numpy(float))

    g = pd.read_csv(DATA / "gap_gw_features.csv").set_index("uid")
    shared = f.loc[nonmag].set_index("uid").index.intersection(g.index)
    gw = g.loc[shared, "gap_gw"]
    gw = gw[np.isfinite(gw)]
    sub = f.set_index("uid").loc[gw.index]
    tasks["gap_gw"] = (sub[feat].apply(pd.to_numeric, errors="coerce").to_numpy(float),
                       gw.to_numpy(float))

    for k, (X, y) in tasks.items():
        X[~np.isfinite(X)] = np.nan          # emass is inf for flat bands
        print(f"{k:8s} X {X.shape}  y {y.shape}  "
              f"median {np.median(y):.2f} eV", flush=True)
    return tasks, feat


def histgb():
    return HistGradientBoostingRegressor(
        random_state=0, loss="absolute_error", max_iter=500, learning_rate=0.06,
        max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=True, n_iter_no_change=25, validation_fraction=0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["cloud", "local"], default="cloud")
    ap.add_argument("--model-version",
                    choices=sorted(tabpfn_backend.MODEL_VERSIONS),
                    default=tabpfn_backend.DEFAULT_VERSION,
                    help="cloud model to pin; 'auto' is not reproducible")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    make_tabpfn = tabpfn_backend.make_regressor(
        args.backend, getattr(args, "model_version", tabpfn_backend.DEFAULT_VERSION))
    print(tabpfn_backend.describe(
        args.backend, getattr(args, "model_version", tabpfn_backend.DEFAULT_VERSION)),
        flush=True)

    tasks, feat = build_tasks()
    print(f"{len(feat)} PBE-level features, HSE-derived columns excluded\n", flush=True)

    out_path = DATA / "task_scaling.csv"
    records, done = [], set()
    if args.resume and out_path.exists():
        records = pd.read_csv(out_path).to_dict("records")
        done = {(r["task"], int(r["fold"]), int(r["n_materials"]), int(r["seed"]))
                for r in records}
        print(f"resuming from {len(records)} rows", flush=True)

    def flush():
        pd.DataFrame(records).to_csv(out_path, index=False)

    for task, (X, y) in tasks.items():
        kf = KFold(args.folds, shuffle=True, random_state=0)
        for fold, (tr, te) in enumerate(kf.split(X)):
            Xte, yte = X[te], y[te]
            for n_mat in GRIDS[task]:
                if n_mat > len(tr):
                    continue
                for seed in range(args.seeds):
                    if (task, fold, n_mat, seed) in done:
                        continue
                    rs = np.random.RandomState(1000 * seed + fold)
                    sel = rs.choice(tr, n_mat, replace=False)
                    Xtr, ytr = X[sel], y[sel]

                    t0 = time.time()
                    tp = make_tabpfn()
                    tp.fit(Xtr, ytr)
                    mae_tp = mean_absolute_error(yte, np.asarray(tp.predict(Xte)))
                    hg = histgb()
                    hg.fit(Xtr, ytr)
                    mae_hg = mean_absolute_error(yte, hg.predict(Xte))

                    records.append({"task": task, "fold": fold, "n_materials": n_mat,
                                    "seed": seed, "n_test": len(te),
                                    "mae_tabpfn": mae_tp, "mae_histgb": mae_hg})
                    flush()
                    print(f"{task:8s} fold {fold:2d}  n {n_mat:5d}  seed {seed}  "
                          f"tabpfn {mae_tp:.4f}  histgb {mae_hg:.4f}  "
                          f"({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(records)
    print("\n=== MAE vs training materials (fold-equal means, eV) ===")
    for task in tasks:
        t = df[df.task == task]
        piv = t.groupby(["fold", "n_materials"])[["mae_tabpfn", "mae_histgb"]].mean()
        g = piv.groupby("n_materials").mean()
        print(f"\n{task}:")
        print(g.round(4).to_string())
        ns = g.index.values.astype(float)
        for col in ("mae_tabpfn", "mae_histgb"):
            b, a = np.polyfit(np.log(ns), np.log(g[col].values), 1)
            pred = np.exp(a) * ns ** b
            r2 = 1 - ((np.log(g[col]) - np.log(pred)) ** 2).sum() / \
                     ((np.log(g[col]) - np.log(g[col]).mean()) ** 2).sum()
            print(f"  {col:11s} exponent {b:+.3f}   log-log R2 {r2:.3f}")

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
