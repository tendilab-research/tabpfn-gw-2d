import tabpfn_backend
"""Phase A-1: GW band-edge prediction and photocatalyst screening.

Step 1 - validate: 5-fold CV (non-magnetic GW materials) for vacuum-referenced
vbm_gw and cbm_gw. Baseline: linear on the HSE edge.
Step 2 - screen: fit on all non-magnetic GW materials, predict median and
[q05, q95] for the screening targets.
Step 3 - discover: overall water-splitting alignment at GW level.
    HER: E(H+/H2)  = -4.44 eV vs vacuum   -> CBM must lie above
    OER: E(O2/H2O) = -5.67 eV vs vacuum   -> VBM must lie below
  relaxed      = medians pass + visible gap (1.23 < Eg < 3.0 eV)
  conservative = 90% CI entirely on the correct side of both levels

Outputs: band_edge_cv.csv, band_edge_release.csv, photocatalyst_candidates.csv
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline

from run_delta_regression import DATA

META = ["uid", "folder", "has_gw", "magnetic", "gap_gw"]
E_HER, E_OER = -4.44, -5.67


def main():
    from tabpfn_client import TabPFNRegressor

    df = pd.read_csv(DATA / "screening_features.csv")
    feature_cols = [c for c in df.columns if c not in META]

    # vbm_gw / cbm_gw are not in screening_features; pull from gap features
    gw_feats = pd.read_csv(DATA / "gap_gw_features.csv").set_index("uid")
    train = df[df.has_gw & ~df.magnetic].copy()
    train["vbm_gw"] = train.uid.map(gw_feats["vbm_gw"])
    train["cbm_gw"] = train.uid.map(gw_feats["cbm_gw"])
    train = train.dropna(subset=["vbm_gw", "cbm_gw"])
    target = df[~df.has_gw & ~df.magnetic & (df.gap_hse > 0.05)].copy()
    print(f"train {len(train)} | screening targets {len(target)}", flush=True)

    def X(d):
        x = d[feature_cols].to_numpy(dtype=float)
        x[~np.isfinite(x)] = np.nan
        return x

    # ---- Step 1: CV validation ----
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    cv_rows = []
    for edge, hse_col in [("vbm_gw", "vbm_hse"), ("cbm_gw", "cbm_hse")]:
        y = train[edge].to_numpy()
        mae_tp, mae_lin, cov90 = [], [], []
        for tr, te in kf.split(train):
            m = tabpfn_backend.make_regressor("cloud", MODEL_VERSION)()
            m.fit(X(train.iloc[tr]), y[tr])
            q = np.asarray(m.predict(X(train.iloc[te]),
                                     output_type="quantiles",
                                     quantiles=[0.05, 0.5, 0.95]))
            mae_tp.append(mean_absolute_error(y[te], q[1]))
            cov90.append(np.mean((y[te] >= q[0]) & (y[te] <= q[2])))
            lin = make_pipeline(SimpleImputer(strategy="median"), LinearRegression())
            lin.fit(train.iloc[tr][[hse_col]], y[tr])
            mae_lin.append(mean_absolute_error(y[te], lin.predict(train.iloc[te][[hse_col]])))
        cv_rows.append({"target": edge,
                        "tabpfn_mae": np.mean(mae_tp), "tabpfn_std": np.std(mae_tp),
                        "linear_hse_mae": np.mean(mae_lin),
                        "coverage90": np.mean(cov90)})
        print(f"{edge}: TabPFN {np.mean(mae_tp):.3f}±{np.std(mae_tp):.3f} eV "
              f"(lin-HSE {np.mean(mae_lin):.3f}), 90% cov {np.mean(cov90):.3f}",
              flush=True)
    pd.DataFrame(cv_rows).to_csv(DATA / "band_edge_cv.csv", index=False)

    # ---- Step 2: screening predictions ----
    out = target[["uid", "folder", "gap", "gap_hse", "vbm_hse", "cbm_hse"]].copy()
    for edge in ["vbm_gw", "cbm_gw"]:
        m = tabpfn_backend.make_regressor("cloud", MODEL_VERSION)()
        m.fit(X(train), train[edge].to_numpy())
        q = np.asarray(m.predict(X(target), output_type="quantiles",
                                 quantiles=[0.05, 0.5, 0.95]))
        pre = edge.replace("_gw", "")
        out[f"{pre}_pred"], out[f"{pre}_q05"], out[f"{pre}_q95"] = q[1], q[0], q[2]
        print(f"predicted {edge} for {len(out)} materials", flush=True)

    out["gap_edges"] = out.cbm_pred - out.vbm_pred
    scr = pd.read_csv(DATA / "screening_release.csv")[["uid", "gap_gw_pred"]]
    out = out.merge(scr, on="uid", how="left")
    print(f"consistency corr(gap_edges, gap_model): "
          f"{out[['gap_edges', 'gap_gw_pred']].corr().iloc[0, 1]:.3f}", flush=True)

    # ---- Step 3: photocatalyst selection ----
    visible = (out.gap_edges > 1.23) & (out.gap_edges < 3.0)
    relaxed = visible & (out.cbm_pred > E_HER) & (out.vbm_pred < E_OER)
    conservative = relaxed & (out.cbm_q05 > E_HER) & (out.vbm_q95 < E_OER)
    out["photocat_relaxed"] = relaxed
    out["photocat_conservative"] = conservative
    out.to_csv(DATA / "band_edge_release.csv", index=False)

    cand = out[relaxed].sort_values("gap_edges")
    cand.to_csv(DATA / "photocatalyst_candidates.csv", index=False)
    print(f"\nphotocatalyst candidates (visible, straddling): "
          f"relaxed {relaxed.sum()}, conservative {conservative.sum()}")
    cols = ["uid", "gap_edges", "vbm_pred", "cbm_pred"]
    print(cand[cols].head(15).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
