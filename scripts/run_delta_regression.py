"""Main experiment: state-level QP correction learning.

    f(material features, state features) -> delta = eps_GW - eps_HSE

2.5M states / 261 materials. Split is ALWAYS by material (GroupKFold 5).
TabPFN trains on a per-material-stratified subsample (its context window is
the 'small data' budget); HistGB also gets a full-train variant as the
strongest classical baseline. Test states are subsampled per material for
metric stability at manageable inference cost.

Backends: --backend cloud (tabpfn-client, latest hosted model, default)
          --backend local (tabpfn 2.0.9 in .venv, CPU)
"""

import argparse
import time
from pathlib import Path

import paths
import tabpfn_backend

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline

DATA = paths.DATA
META_COLS = ["uid", "folder", "has_hse_json"]
TARGET_COLS = ["gap_gw", "gap_dir_gw", "vbm_gw", "cbm_gw"]
STATE_COLS = ["eps_hse", "eps_rel_ef", "band_rel", "occupied",
              "kx", "ky", "k_norm", "path_frac"]
RNG = np.random.RandomState(0)


def load_all():
    z = np.load(DATA / "delta_eps_dataset.npz", allow_pickle=True)
    uid = z["uid"]
    state_X = np.stack([z[c].astype(np.float32) for c in STATE_COLS], axis=1)
    y = z["delta"].astype(np.float32)

    feats = pd.read_csv(DATA / "gap_gw_features.csv")
    mat_cols = [c for c in feats.columns if c not in META_COLS + TARGET_COLS]
    feats = feats.set_index("uid")
    mats = sorted(set(uid.tolist()))
    mat_X = feats.loc[mats, mat_cols].to_numpy(dtype=np.float32)
    mat_X[~np.isfinite(mat_X)] = np.nan

    mat_index = {u: i for i, u in enumerate(mats)}
    row_mat = np.array([mat_index[u] for u in uid.tolist()], dtype=np.int32)
    return state_X, y, row_mat, np.array(mats), mat_X, mat_cols


def subsample_per_material(row_mat, rows_idx, per_mat, rng):
    take = []
    for m in np.unique(row_mat[rows_idx]):
        cand = rows_idx[row_mat[rows_idx] == m]
        take.append(rng.choice(cand, size=min(per_mat, len(cand)), replace=False))
    return np.concatenate(take)


def build_X(idx, state_X, row_mat, mat_X):
    return np.hstack([state_X[idx], mat_X[row_mat[idx]]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["cloud", "local"], default="cloud")
    ap.add_argument("--model-version",
                    choices=sorted(tabpfn_backend.MODEL_VERSIONS),
                    default=tabpfn_backend.DEFAULT_VERSION,
                    help="cloud model to pin; 'auto' is not reproducible")
    ap.add_argument("--train-rows", type=int, default=10000)
    ap.add_argument("--test-per-mat", type=int, default=100)
    ap.add_argument("--drop-magnetic", action="store_true",
                    help="exclude magnetic materials (magnetic_flags.csv)")
    ap.add_argument("--no-hse-mat-cols", action="store_true",
                    help="drop HSE scalars from material features")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    ap.add_argument("--embeddings", default=None,
                    help="npz with {uid, emb}; appended to material features "
                         "(PCA-reduced to --emb-dims)")
    ap.add_argument("--emb-dims", type=int, default=64)
    ap.add_argument("--spin-features", action="store_true",
                    help="append spin_features.csv columns to material features")
    args = ap.parse_args()

    make_tabpfn = tabpfn_backend.make_regressor(
        args.backend, getattr(args, "model_version", tabpfn_backend.DEFAULT_VERSION))
    print(tabpfn_backend.describe(
        args.backend, getattr(args, "model_version", tabpfn_backend.DEFAULT_VERSION)),
        flush=True)

    state_X, y, row_mat, mats, mat_X, mat_cols = load_all()

    if args.no_hse_mat_cols:
        hse_mat = ["gap_hse", "gap_dir_hse", "vbm_hse", "cbm_hse", "efermi_hse"]
        keep = [i for i, c in enumerate(mat_cols) if c not in hse_mat]
        mat_X, mat_cols = mat_X[:, keep], [mat_cols[i] for i in keep]

    if args.spin_features:
        sf = pd.read_csv(DATA / "spin_features.csv").set_index("uid")
        spin_cols = [c for c in sf.columns]
        spin_X = sf.loc[mats.tolist()].to_numpy(dtype=np.float32)
        mat_X = np.hstack([mat_X, spin_X])
        mat_cols = mat_cols + spin_cols

    if args.embeddings:
        ez = np.load(args.embeddings, allow_pickle=True)
        emb_map = {u: e for u, e in zip(ez["uid"].tolist(), ez["emb"])}
        emb = np.stack([emb_map[u] for u in mats]).astype(np.float32)
        # PCA to keep the total feature count TabPFN-friendly
        emb = emb - emb.mean(axis=0)
        _, _, vt = np.linalg.svd(emb, full_matrices=False)
        emb = emb @ vt[: args.emb_dims].T
        mat_X = np.hstack([mat_X, emb])
        mat_cols = mat_cols + [f"mace_{i}" for i in range(emb.shape[1])]

    if args.drop_magnetic:
        flags = pd.read_csv(DATA / "magnetic_flags.csv", index_col=0)
        keep_mat = np.array([not flags.loc[u, "magnetic"] for u in mats])
        keep_idx = np.flatnonzero(keep_mat)
        remap = -np.ones(len(mats), dtype=np.int32)
        remap[keep_idx] = np.arange(len(keep_idx))
        keep_rows = np.isin(row_mat, keep_idx)
        state_X, y = state_X[keep_rows], y[keep_rows]
        row_mat = remap[row_mat[keep_rows]]
        mats, mat_X = mats[keep_idx], mat_X[keep_idx]

    n_feat = state_X.shape[1] + mat_X.shape[1]
    print(f"{len(y):,} states, {len(mats)} materials, {n_feat} features, "
          f"backend={args.backend}, drop_magnetic={args.drop_magnetic}, "
          f"no_hse_mat_cols={args.no_hse_mat_cols}", flush=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)  # material-level folds
    fold_metrics = {}   # model -> list of (mae, rmse, mae_val, mae_cond)
    oof_records = []

    for fold, (tr_mat_idx, te_mat_idx) in enumerate(kf.split(mats)):
        tr_rows = np.flatnonzero(np.isin(row_mat, tr_mat_idx))
        te_rows = np.flatnonzero(np.isin(row_mat, te_mat_idx))

        per_mat_train = max(1, args.train_rows // len(tr_mat_idx))
        tr_sub = subsample_per_material(row_mat, tr_rows, per_mat_train, RNG)
        te_sub = subsample_per_material(row_mat, te_rows, args.test_per_mat, RNG)

        Xtr, ytr = build_X(tr_sub, state_X, row_mat, mat_X), y[tr_sub]
        Xte, yte = build_X(te_sub, state_X, row_mat, mat_X), y[te_sub]
        occ_te = state_X[te_sub, STATE_COLS.index("occupied")] > 0.5

        # classical baselines
        preds = {}
        preds["zero"] = np.zeros(len(yte))

        occ_tr = state_X[tr_rows, STATE_COLS.index("occupied")] > 0.5
        scis_v, scis_c = y[tr_rows][occ_tr].mean(), y[tr_rows][~occ_tr].mean()
        preds["scissors"] = np.where(occ_te, scis_v, scis_c)

        lin = make_pipeline(SimpleImputer(strategy="median"), LinearRegression())
        lin.fit(Xtr, ytr)
        preds["linear_sub"] = lin.predict(Xte)

        hg = HistGradientBoostingRegressor(random_state=0)
        hg.fit(Xtr, ytr)
        preds["histgb_sub"] = hg.predict(Xte)

        t0 = time.time()
        hg_full = HistGradientBoostingRegressor(random_state=0)
        Xtr_full = build_X(tr_rows, state_X, row_mat, mat_X)
        hg_full.fit(Xtr_full, y[tr_rows])
        preds["histgb_full"] = hg_full.predict(Xte)
        del Xtr_full
        t_hgf = time.time() - t0

        t0 = time.time()
        tp = make_tabpfn()
        tp.fit(Xtr, ytr)
        preds["tabpfn"] = np.asarray(tp.predict(Xte))
        t_tp = time.time() - t0

        for name, p in preds.items():
            mae = mean_absolute_error(yte, p)
            rmse = float(np.sqrt(mean_squared_error(yte, p)))
            mae_v = mean_absolute_error(yte[occ_te], p[occ_te])
            mae_c = mean_absolute_error(yte[~occ_te], p[~occ_te])
            fold_metrics.setdefault(name, []).append((mae, rmse, mae_v, mae_c))
        oof_records.append({"fold": fold, "test_idx": te_sub,
                            "pred_tabpfn": preds["tabpfn"]})
        f = {k: v[-1][0] for k, v in fold_metrics.items()}
        print(f"fold {fold}: " + "  ".join(f"{k} {v:.3f}" for k, v in f.items())
              + f"  [train {len(tr_sub):,}, test {len(te_sub):,}; "
                f"histgb_full {t_hgf:.0f}s, tabpfn {t_tp:.0f}s]", flush=True)

    print("\n=== Δε MAE over 5 folds (eV) ===")
    rows = []
    for name, vals in fold_metrics.items():
        v = np.array(vals)
        rows.append({"model": name,
                     "mae": v[:, 0].mean(), "mae_std": v[:, 0].std(),
                     "rmse": v[:, 1].mean(),
                     "mae_valence": v[:, 2].mean(), "mae_conduction": v[:, 3].mean()})
        print(f"{name:12s} MAE {v[:,0].mean():.3f}±{v[:,0].std():.3f}  "
              f"RMSE {v[:,1].mean():.3f}  val {v[:,2].mean():.3f}  "
              f"cond {v[:,3].mean():.3f}")
    suffix = args.backend + (f"_{args.tag}" if args.tag else "")
    pd.DataFrame(rows).to_csv(DATA / f"delta_cv_results_{suffix}.csv", index=False)

    np.savez_compressed(DATA / f"delta_oof_{suffix}.npz",
                        test_idx=np.concatenate([r["test_idx"] for r in oof_records]),
                        pred=np.concatenate([r["pred_tabpfn"] for r in oof_records]))
    print("saved results")


if __name__ == "__main__":
    main()
