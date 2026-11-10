import tabpfn_backend
"""GW-gap screening release: predict G0W0 gaps (+calibrated intervals) for
all non-magnetic 2D semiconductors that have HSE data but no GW calculation.

Train: 319 non-magnetic GW-labelled materials (all features).
Predict: median and [q05, q95] for 2,613 screening targets.

Output: data/screening_release.csv
"""

import numpy as np
import pandas as pd

from run_delta_regression import DATA

META = ["uid", "folder", "has_gw", "magnetic", "gap_gw"]


def main():
    from tabpfn_client import TabPFNRegressor

    df = pd.read_csv(DATA / "screening_features.csv")
    feature_cols = [c for c in df.columns if c not in META]

    train = df[df.has_gw & ~df.magnetic & df.gap_gw.notna()]
    target = df[~df.has_gw & ~df.magnetic & (df.gap_hse > 0.05)]
    print(f"train {len(train)} | screening targets {len(target)}")

    def X(d):
        x = d[feature_cols].to_numpy(dtype=float)
        x[~np.isfinite(x)] = np.nan
        return x

    m = tabpfn_backend.make_regressor("cloud", MODEL_VERSION)()
    m.fit(X(train), train.gap_gw.to_numpy())
    q = np.asarray(m.predict(X(target), output_type="quantiles",
                             quantiles=[0.05, 0.5, 0.95]))

    out = target[["uid", "folder", "gap", "gap_hse"]].copy()
    out["gap_gw_pred"] = q[1]
    out["gap_gw_q05"] = q[0]
    out["gap_gw_q95"] = q[2]
    out["ci90_width"] = q[2] - q[0]
    out = out.sort_values("gap_gw_pred")
    out.to_csv(DATA / "screening_release.csv", index=False)

    print(f"saved {len(out)} predictions -> screening_release.csv")
    print(f"pred gap_gw: median {out.gap_gw_pred.median():.2f}, "
          f"range [{out.gap_gw_pred.min():.2f}, {out.gap_gw_pred.max():.2f}] eV")
    print(f"90% CI width: median {out.ci90_width.median():.3f} eV, "
          f"p90 {out.ci90_width.quantile(0.9):.3f} eV")
    frac_above = (out.gap_gw_pred > out.gap_hse).mean()
    print(f"sanity - predicted GW gap > HSE gap: {frac_above:.1%} "
          f"(GW should open the gap)")


if __name__ == "__main__":
    main()
