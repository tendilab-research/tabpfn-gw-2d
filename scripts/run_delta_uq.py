import tabpfn_backend
"""UQ calibration for the QP-correction model (non-magnetic scope).

Same material-level 5-fold CV as the main experiment, but TabPFN predicts
quantiles. We test whether the predictive distribution is calibrated:
empirical coverage of central intervals vs nominal, and per-level
reliability P(y <= q_level).

Output: data/delta_uq_calibration.npz + printed summary.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from run_delta_regression import (DATA, STATE_COLS, load_all,
                                  subsample_per_material, build_X)

QUANTS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
RNG = np.random.RandomState(0)


def main():
    from tabpfn_client import TabPFNRegressor

    state_X, y, row_mat, mats, mat_X, mat_cols = load_all()

    flags = pd.read_csv(DATA / "magnetic_flags.csv", index_col=0)
    keep_idx = np.flatnonzero(np.array([not flags.loc[u, "magnetic"] for u in mats]))
    remap = -np.ones(len(mats), dtype=np.int32)
    remap[keep_idx] = np.arange(len(keep_idx))
    keep_rows = np.isin(row_mat, keep_idx)
    state_X, y = state_X[keep_rows], y[keep_rows]
    row_mat = remap[row_mat[keep_rows]]
    mats, mat_X = mats[keep_idx], mat_X[keep_idx]
    print(f"{len(y):,} states, {len(mats)} non-magnetic materials", flush=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    all_q, all_y = [], []
    for fold, (tr_mat_idx, te_mat_idx) in enumerate(kf.split(mats)):
        tr_rows = np.flatnonzero(np.isin(row_mat, tr_mat_idx))
        te_rows = np.flatnonzero(np.isin(row_mat, te_mat_idx))
        tr_sub = subsample_per_material(row_mat, tr_rows,
                                        max(1, 10000 // len(tr_mat_idx)), RNG)
        te_sub = subsample_per_material(row_mat, te_rows, 100, RNG)

        m = tabpfn_backend.make_regressor("cloud", MODEL_VERSION)()
        m.fit(build_X(tr_sub, state_X, row_mat, mat_X), y[tr_sub])
        q = np.asarray(m.predict(build_X(te_sub, state_X, row_mat, mat_X),
                                 output_type="quantiles", quantiles=QUANTS))
        all_q.append(q)
        all_y.append(y[te_sub])
        cov90 = np.mean((y[te_sub] >= q[0]) & (y[te_sub] <= q[-1]))
        print(f"fold {fold}: n={len(te_sub):,}  90% coverage {cov90:.3f}", flush=True)

    q = np.concatenate(all_q, axis=1)
    yy = np.concatenate(all_y)

    print("\n=== calibration (pooled) ===")
    print("central interval coverage:")
    for lo, hi, nom in [(0, 6, 0.90), (1, 5, 0.80), (2, 4, 0.50)]:
        cov = np.mean((yy >= q[lo]) & (yy <= q[hi]))
        print(f"  nominal {nom:.0%}: empirical {cov:.3f}")
    print("per-level reliability P(y <= q_level):")
    for i, ql in enumerate(QUANTS):
        print(f"  {ql:.2f}: {np.mean(yy <= q[i]):.3f}")
    w90 = q[-1] - q[0]
    print(f"90% interval width: median {np.median(w90):.3f} eV, "
          f"p90 {np.percentile(w90, 90):.3f} eV")

    np.savez_compressed(DATA / "delta_uq_calibration.npz",
                        quantile_levels=np.array(QUANTS), q=q, y=yy)
    print("saved", DATA / "delta_uq_calibration.npz")


if __name__ == "__main__":
    main()
