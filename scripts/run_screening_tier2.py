import tabpfn_backend
"""Phase A-2: extend the screening release to PBE-only materials (tier 2).

Tier 2 = non-magnetic, no GW, no HSE data, PBE gap > 0.05 eV (921 materials).
Models: TabPFN trained on the 319 non-magnetic GW materials using the
feature set WITHOUT the five HSE scalars (CV: gap 0.099 eV).
Predicts gap_gw, vbm_gw, cbm_gw with [q05, q50, q95].

Merges tier 1 (band_edge_release.csv, HSE-informed) into the combined
data/screening_release_full.csv with uniform photocatalyst flags.
"""

import numpy as np
import pandas as pd

from run_delta_regression import DATA

META = ["uid", "folder", "has_gw", "magnetic", "gap_gw"]
HSE_COLS = ["gap_hse", "gap_dir_hse", "vbm_hse", "cbm_hse", "efermi_hse"]
E_HER, E_OER = -4.44, -5.67


def main():
    from tabpfn_client import TabPFNRegressor

    df = pd.read_csv(DATA / "screening_features_all.csv")
    feature_cols = [c for c in df.columns
                    if c not in META and c not in HSE_COLS]

    gw_feats = pd.read_csv(DATA / "gap_gw_features.csv").set_index("uid")
    train = df[df.has_gw & ~df.magnetic].copy()
    for t in ["vbm_gw", "cbm_gw"]:
        train[t] = train.uid.map(gw_feats[t])
    train["gap_gw"] = train.uid.map(gw_feats["gap_gw"])
    train = train.dropna(subset=["gap_gw", "vbm_gw", "cbm_gw"])

    tier2 = df[~df.has_gw & ~df.magnetic & df.gap_hse.isna()
               & (df.gap > 0.05)].copy()
    print(f"train {len(train)} | tier-2 targets {len(tier2)} "
          f"| features {len(feature_cols)} (no HSE)", flush=True)

    def X(d):
        x = d[feature_cols].to_numpy(dtype=float)
        x[~np.isfinite(x)] = np.nan
        return x

    out2 = tier2[["uid", "folder", "gap"]].copy()
    for target_col, pre in [("gap_gw", "gap_gw"), ("vbm_gw", "vbm"),
                            ("cbm_gw", "cbm")]:
        m = tabpfn_backend.make_regressor("cloud", MODEL_VERSION)()
        m.fit(X(train), train[target_col].to_numpy())
        q = np.asarray(m.predict(X(tier2), output_type="quantiles",
                                 quantiles=[0.05, 0.5, 0.95]))
        out2[f"{pre}_pred"], out2[f"{pre}_q05"], out2[f"{pre}_q95"] = q[1], q[0], q[2]
        print(f"tier-2 {target_col} predicted", flush=True)
    out2["tier"] = 2

    # tier 1: HSE-informed release (gap from run_screening, edges from run_band_edges)
    t1 = pd.read_csv(DATA / "band_edge_release.csv")
    g1 = pd.read_csv(DATA / "screening_release.csv")[
        ["uid", "gap_gw_pred", "gap_gw_q05", "gap_gw_q95"]]
    t1 = t1.drop(columns=["gap_gw_pred"]).merge(g1, on="uid")
    t1["tier"] = 1

    common = ["uid", "folder", "tier", "gap_gw_pred", "gap_gw_q05", "gap_gw_q95",
              "vbm_pred", "vbm_q05", "vbm_q95", "cbm_pred", "cbm_q05", "cbm_q95"]
    full = pd.concat([t1.reindex(columns=common + ["gap", "gap_hse"]),
                      out2.reindex(columns=common + ["gap"])], ignore_index=True)

    gap_edges = full.cbm_pred - full.vbm_pred
    visible = (gap_edges > 1.23) & (gap_edges < 3.0)
    full["photocat_relaxed"] = visible & (full.cbm_pred > E_HER) & (full.vbm_pred < E_OER)
    full["photocat_conservative"] = (full.photocat_relaxed
                                     & (full.cbm_q05 > E_HER) & (full.vbm_q95 < E_OER))
    full.to_csv(DATA / "screening_release_full.csv", index=False)

    print(f"\ncombined release: {len(full)} materials "
          f"(tier1 {(full.tier == 1).sum()}, tier2 {(full.tier == 2).sum()})")
    print(f"90% CI width (gap): tier1 median "
          f"{(t1.gap_gw_q95 - t1.gap_gw_q05).median():.3f}, tier2 median "
          f"{(out2.gap_gw_q95 - out2.gap_gw_q05).median():.3f} eV")
    for flag in ["photocat_relaxed", "photocat_conservative"]:
        n1 = full[(full.tier == 1) & full[flag]].shape[0]
        n2 = full[(full.tier == 2) & full[flag]].shape[0]
        print(f"{flag}: {n1 + n2} (tier1 {n1}, tier2 {n2})")


if __name__ == "__main__":
    main()
