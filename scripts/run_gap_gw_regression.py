"""gap_gw regression milestone: TabPFN vs baselines on 339 GW materials.

The question this answers: does TabPFN beat the classic literature baseline
(linear fit of gap_gw on gap_hse) and standard tabular models, on genuinely
small materials data?

Protocol: 5-fold CV x 4 repeats (20 fits per model), same splits for all
models. Metrics: MAE / RMSE / R^2, mean +- std over folds.
"""

import argparse
import time
from pathlib import Path

import paths
import tabpfn_backend

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = paths.DATA
META_COLS = ["uid", "folder", "has_hse_json"]
TARGET_COLS = ["gap_gw", "gap_dir_gw", "vbm_gw", "cbm_gw"]
HSE_COLS = ["gap_hse", "gap_dir_hse", "vbm_hse", "cbm_hse", "efermi_hse"]


def get_models(feature_cols, backend="local",
               model_version=tabpfn_backend.DEFAULT_VERSION):
    make_tabpfn = tabpfn_backend.make_regressor(backend, model_version)
    print(tabpfn_backend.describe(backend, model_version), flush=True)

    no_hse = [c for c in feature_cols if c not in HSE_COLS]
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "lin_gap_hse": (make_pipeline(imp(), LinearRegression()), ["gap_hse"]),
        "lin_gap_pbe": (make_pipeline(imp(), LinearRegression()), ["gap"]),
        "ridge_all": (make_pipeline(imp(), StandardScaler(), Ridge(alpha=1.0)), feature_cols),
        "rf_all": (make_pipeline(imp(), RandomForestRegressor(500, random_state=0, n_jobs=-1)), feature_cols),
        "histgb_all": (HistGradientBoostingRegressor(random_state=0), feature_cols),
        "tabpfn_all": (make_tabpfn(), feature_cols),
        "tabpfn_no_hse": (make_tabpfn(), no_hse),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["local", "cloud"], default="local")
    ap.add_argument("--model-version",
                    choices=sorted(tabpfn_backend.MODEL_VERSIONS),
                    default=tabpfn_backend.DEFAULT_VERSION,
                    help="cloud model to pin; 'auto' is not reproducible")
    ap.add_argument("--drop-magnetic", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    df = pd.read_csv(DATA / "gap_gw_features.csv")
    if args.drop_magnetic:
        flags = pd.read_csv(DATA / "magnetic_flags.csv", index_col=0)["magnetic"]
        sf = pd.read_csv(DATA / "spin_features.csv").set_index("uid")["is_magnetic"]
        magnetic = df["uid"].map(flags).fillna(df["uid"].map(sf) > 0).astype(bool)
        df = df[~magnetic].reset_index(drop=True)
    y = df["gap_gw"].to_numpy()
    feature_cols = [c for c in df.columns if c not in META_COLS + TARGET_COLS]
    print(f"{len(df)} materials, {len(feature_cols)} features, "
          f"drop_magnetic={args.drop_magnetic}, backend={args.backend}")

    cv = RepeatedKFold(n_splits=5, n_repeats=4, random_state=0)
    splits = list(cv.split(df))
    models = get_models(feature_cols, args.backend, args.model_version)

    results = {}
    oof = {}  # out-of-fold predictions from repeat 0 (first 5 folds)
    for name, (model, cols) in models.items():
        X = df[cols].to_numpy(dtype=float)
        X[~np.isfinite(X)] = np.nan  # emass is inf for flat bands
        mae, rmse, r2 = [], [], []
        oof_pred = np.full(len(df), np.nan)
        t0 = time.time()
        for i, (tr, te) in enumerate(splits):
            from sklearn.base import clone
            m = clone(model)
            m.fit(X[tr], y[tr])
            p = m.predict(X[te])
            mae.append(mean_absolute_error(y[te], p))
            rmse.append(np.sqrt(mean_squared_error(y[te], p)))
            r2.append(r2_score(y[te], p))
            if i < 5:
                oof_pred[te] = p
        results[name] = (np.mean(mae), np.std(mae), np.mean(rmse), np.std(rmse),
                         np.mean(r2), np.std(r2))
        oof[name] = oof_pred
        print(f"{name:15s} MAE {np.mean(mae):.3f}±{np.std(mae):.3f}  "
              f"RMSE {np.mean(rmse):.3f}±{np.std(rmse):.3f}  "
              f"R2 {np.mean(r2):.3f}±{np.std(r2):.3f}  ({time.time()-t0:.0f}s)",
              flush=True)

    suffix = ("_" + args.tag) if args.tag else ""
    pd.DataFrame(results, index=["mae", "mae_std", "rmse", "rmse_std", "r2", "r2_std"]).T \
        .to_csv(DATA / f"gap_gw_cv_results{suffix}.csv")
    np.savez(DATA / f"gap_gw_oof{suffix}.npz", y=y, uid=df["uid"].to_numpy(),
             **{f"pred_{k}": v for k, v in oof.items()})
    print("saved:", DATA / f"gap_gw_cv_results{suffix}.csv")


if __name__ == "__main__":
    main()
