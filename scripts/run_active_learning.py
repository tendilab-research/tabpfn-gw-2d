import tabpfn_backend
"""Phase A-4: active-learning simulation on the gap_gw task (non-magnetic).

Question: if you were building a GW database from scratch with a fixed
budget, how many G0W0 calculations do you need - and does TabPFN's
uncertainty tell you WHICH material to compute next?

Protocol (per seed): hold out 64 test materials; start with 30 labelled,
acquire in batches of 20 up to 250 using either
  random      - uniform sampling from the unlabelled pool
  uncertainty - widest 90% CI on the unlabelled pool
Evaluate test MAE after each round. 5 seeds.

Output: data/active_learning.csv
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from run_delta_regression import DATA

META_COLS = ["uid", "folder", "has_hse_json"]
TARGET_COLS = ["gap_gw", "gap_dir_gw", "vbm_gw", "cbm_gw"]
START, BATCH, MAX_LABELS, SEEDS = 30, 20, 250, 5


def main():
    from tabpfn_client import TabPFNRegressor

    df = pd.read_csv(DATA / "gap_gw_features.csv")
    flags = pd.read_csv(DATA / "spin_features.csv").set_index("uid")["is_magnetic"]
    df = df[df.uid.map(flags) == 0].reset_index(drop=True)
    y = df.gap_gw.to_numpy()
    feature_cols = [c for c in df.columns if c not in META_COLS + TARGET_COLS]
    X = df[feature_cols].to_numpy(dtype=float)
    X[~np.isfinite(X)] = np.nan

    rows = []
    for seed in range(SEEDS):
        pool_idx, test_idx = train_test_split(np.arange(len(df)), test_size=64,
                                              random_state=seed)
        rng = np.random.RandomState(seed)
        init = rng.choice(pool_idx, size=START, replace=False)
        for strategy in ["random", "uncertainty"]:
            labelled = list(init)
            unlabelled = [i for i in pool_idx if i not in set(labelled)]
            while True:
                m = tabpfn_backend.make_regressor("cloud", MODEL_VERSION)()
                m.fit(X[labelled], y[labelled])
                q_test = np.asarray(m.predict(X[test_idx],
                                              output_type="quantiles",
                                              quantiles=[0.05, 0.5, 0.95]))
                mae = mean_absolute_error(y[test_idx], q_test[1])
                rows.append({"seed": seed, "strategy": strategy,
                             "n_labels": len(labelled), "test_mae": mae})
                print(f"seed {seed} {strategy:11s} n={len(labelled):3d} "
                      f"MAE {mae:.3f}", flush=True)
                if len(labelled) >= MAX_LABELS or not unlabelled:
                    break
                if strategy == "random":
                    pick = rng.choice(len(unlabelled),
                                      size=min(BATCH, len(unlabelled)),
                                      replace=False)
                else:
                    q_pool = np.asarray(m.predict(X[unlabelled],
                                                  output_type="quantiles",
                                                  quantiles=[0.05, 0.5, 0.95]))
                    width = q_pool[2] - q_pool[0]
                    pick = np.argsort(width)[-min(BATCH, len(unlabelled)):]
                chosen = [unlabelled[i] for i in pick]
                labelled.extend(chosen)
                unlabelled = [i for i in unlabelled if i not in set(chosen)]

    res = pd.DataFrame(rows)
    res.to_csv(DATA / "active_learning.csv", index=False)
    print("\n=== mean test MAE by n_labels ===")
    piv = res.groupby(["strategy", "n_labels"]).test_mae.mean().unstack(0)
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
