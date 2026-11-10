"""Does the gradient-boosting baseline actually lose by 32%, or was it just untuned?

The reported comparison uses HistGradientBoostingRegressor at scikit-learn
defaults, whose loss is squared_error, while the reported metric is MAE. A
baseline optimised for the wrong loss makes the headline margin look larger
than it is. This script re-runs the same folds and the same subsamples as
run_delta_regression.py, changing only the baseline configuration, and leaves
every committed result file untouched.

Writes data/baseline_tuning.csv. TabPFN is not called, so no backend is needed.

    python scripts/run_baseline_tuning.py --drop-magnetic
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
import run_delta_regression as R  # noqa: E402  shared library

DATA = R.DATA

# Same configurations, once for the 10k-row context and once for the full fold.
CONFIGS = {
    "default_squared": dict(random_state=0),
    "absolute_loss": dict(random_state=0, loss="absolute_error"),
    "tuned": dict(random_state=0, loss="absolute_error", max_iter=500,
                  learning_rate=0.06, max_leaf_nodes=63,
                  min_samples_leaf=40, l2_regularization=1.0,
                  early_stopping=True, n_iter_no_change=25,
                  validation_fraction=0.1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-rows", type=int, default=10000)
    ap.add_argument("--test-per-mat", type=int, default=100)
    ap.add_argument("--drop-magnetic", action="store_true")
    args = ap.parse_args()

    state_X, y, row_mat, mats, mat_X, mat_cols = R.load_all()

    if args.drop_magnetic:
        flags = pd.read_csv(DATA / "magnetic_flags.csv", index_col=0)
        keep_idx = np.flatnonzero(
            np.array([not flags.loc[u, "magnetic"] for u in mats]))
        remap = -np.ones(len(mats), dtype=np.int32)
        remap[keep_idx] = np.arange(len(keep_idx))
        keep_rows = np.isin(row_mat, keep_idx)
        state_X, y = state_X[keep_rows], y[keep_rows]
        row_mat = remap[row_mat[keep_rows]]
        mats, mat_X = mats[keep_idx], mat_X[keep_idx]

    print(f"{len(y):,} states, {len(mats)} materials, "
          f"{state_X.shape[1] + mat_X.shape[1]} features", flush=True)

    # Identical stream to run_delta_regression.py: same seed, same call order.
    rng = np.random.RandomState(0)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = {}

    for fold, (tr_mat_idx, te_mat_idx) in enumerate(kf.split(mats)):
        tr_rows = np.flatnonzero(np.isin(row_mat, tr_mat_idx))
        te_rows = np.flatnonzero(np.isin(row_mat, te_mat_idx))

        per_mat_train = max(1, args.train_rows // len(tr_mat_idx))
        tr_sub = R.subsample_per_material(row_mat, tr_rows, per_mat_train, rng)
        te_sub = R.subsample_per_material(row_mat, te_rows, args.test_per_mat, rng)

        Xtr, ytr = R.build_X(tr_sub, state_X, row_mat, mat_X), y[tr_sub]
        Xte, yte = R.build_X(te_sub, state_X, row_mat, mat_X), y[te_sub]
        Xtr_full, ytr_full = R.build_X(tr_rows, state_X, row_mat, mat_X), y[tr_rows]

        line = []
        for cfg_name, kw in CONFIGS.items():
            for scope, (Xa, ya) in (("sub", (Xtr, ytr)),
                                    ("full", (Xtr_full, ytr_full))):
                t0 = time.time()
                m = HistGradientBoostingRegressor(**kw)
                m.fit(Xa, ya)
                mae = mean_absolute_error(yte, m.predict(Xte))
                key = f"{cfg_name}|{scope}"
                scores.setdefault(key, []).append(mae)
                line.append(f"{key} {mae:.3f} ({time.time()-t0:.0f}s)")
        del Xtr_full
        print(f"fold {fold}: " + "  ".join(line), flush=True)

    rows = []
    for key, vals in scores.items():
        cfg, scope = key.split("|")
        v = np.array(vals)
        rec = {"config": cfg, "train_scope": scope,
               "n_train_rows": len(tr_sub) if scope == "sub" else len(tr_rows),
               "mae": v.mean(), "mae_std": v.std()}
        rec.update({f"fold{i}": val for i, val in enumerate(v)})
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "baseline_tuning.csv", index=False)

    print("\n=== HistGB baseline, 5 material-level folds (MAE eV) ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    tabpfn_mae = 0.1009  # committed: delta_cv_results_cloud_nomag.csv
    best = out.mae.min()
    worst_default = out[(out.config == "default_squared")].mae.min()
    print(f"\nTabPFN (committed)         {tabpfn_mae:.4f}")
    print(f"best baseline here         {best:.4f}  -> margin {100*(1-tabpfn_mae/best):.1f}%")
    print(f"best default baseline      {worst_default:.4f}  -> margin "
          f"{100*(1-tabpfn_mae/worst_default):.1f}%")
    print("\n=== paired: does 185x more rows help this baseline? ===")
    fcols = [c for c in out.columns if c.startswith("fold")]
    for cfg in CONFIGS:
        sub = out[(out.config == cfg) & (out.train_scope == "sub")][fcols].to_numpy()[0]
        ful = out[(out.config == cfg) & (out.train_scope == "full")][fcols].to_numpy()[0]
        d = sub - ful  # positive = full is better
        try:
            from scipy.stats import ttest_rel
            p = ttest_rel(sub, ful).pvalue
        except Exception:
            p = float("nan")
        print(f"  {cfg:16s} mean gain {d.mean():+.4f} eV   folds favouring full {int((d>0).sum())}/5   paired p = {p:.3f}")

    print(f"\nwrote {DATA/'baseline_tuning.csv'}")


if __name__ == "__main__":
    main()
