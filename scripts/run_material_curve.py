import tabpfn_backend
"""Is the binding constraint the number of materials, or the number of rows?

The paper argues that labels arrive one material at a time, so the effective
sample size is the material count rather than the 2.5M state rows. That claim
has so far been inferred from a model comparison. This measures it directly:
the training context is held at a fixed number of ROWS while the number of
MATERIALS those rows are drawn from is varied. If materials are the binding
constraint the curve keeps falling; if rows are what matter it flattens early.

Also runs the leakage control the claim's other half needs: an otherwise
identical fit whose train/test split is taken over rows instead of materials.

Writes data/material_curve.csv incrementally, so partial runs are usable.

    python scripts/run_material_curve.py --drop-magnetic --backend cloud
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
GRID = [10, 20, 40, 80, 120, 160, 192, 216]  # materials in the training context
# points larger than a fold's training set are skipped automatically, so the
# same grid serves 5-fold (max 192) and 10-fold (max ~216)


def fixed_row_budget(row_mat, rows, mat_sel, budget, rs):
    """Draw ~budget rows spread evenly over mat_sel, whatever len(mat_sel) is."""
    per_mat = max(1, budget // len(mat_sel))
    sub = R.subsample_per_material(row_mat, rows, per_mat, rs)
    if len(sub) > budget:
        sub = rs.choice(sub, budget, replace=False)
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["cloud", "local"], default="cloud")
    ap.add_argument("--model-version",
                    choices=sorted(tabpfn_backend.MODEL_VERSIONS),
                    default=tabpfn_backend.DEFAULT_VERSION,
                    help="cloud model to pin; 'auto' is not reproducible")
    ap.add_argument("--train-rows", type=int, default=10000)
    ap.add_argument("--test-per-mat", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--drop-magnetic", action="store_true")
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--folds", type=int, default=5,
                    help="material-level CV folds; 10 raises power 0.71 -> 0.98")
    ap.add_argument("--tag", default="",
                    help="output suffix, so a re-run never overwrites the 5-fold file")
    ap.add_argument("--resume", action="store_true",
                    help="keep rows already in the output file and skip that work. "
                         "The cloud backend has a daily cell quota; a 429 mid-run "
                         "should not cost you the folds already paid for.")
    args = ap.parse_args()

    make_tabpfn = tabpfn_backend.make_regressor(
        args.backend, getattr(args, "model_version", tabpfn_backend.DEFAULT_VERSION))
    print(tabpfn_backend.describe(
        args.backend, getattr(args, "model_version", tabpfn_backend.DEFAULT_VERSION)),
        flush=True)

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
          f"budget {args.train_rows:,} rows held fixed, backend={args.backend}",
          flush=True)

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=0)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = DATA / f"material_curve{suffix}.csv"
    records, done = [], set()
    if args.resume and out_path.exists():
        prev = pd.read_csv(out_path)
        records = prev.to_dict("records")
        done = {(r["split"], int(r["fold"]), int(r["n_materials"]), int(r["seed"]))
                for r in records}
        print(f"resuming: {len(records)} rows already present, skipping them",
              flush=True)

    def flush():
        pd.DataFrame(records).to_csv(out_path, index=False)

    for fold, (tr_mat_idx, te_mat_idx) in enumerate(kf.split(mats)):
        tr_rows = np.flatnonzero(np.isin(row_mat, tr_mat_idx))
        te_rows = np.flatnonzero(np.isin(row_mat, te_mat_idx))

        # Test set is fixed within a fold so every point on the curve is
        # scored on exactly the same states.
        rs_te = np.random.RandomState(9000 + fold)
        te_sub = R.subsample_per_material(row_mat, te_rows, args.test_per_mat, rs_te)
        Xte, yte = R.build_X(te_sub, state_X, row_mat, mat_X), y[te_sub]

        for n_mat in GRID:
            if n_mat > len(tr_mat_idx):
                continue
            # Every point gets the same number of seeds. When n_mat equals the
            # fold's training-material count the draw is deterministic, so the
            # extra seeds are redundant - but keeping them equal is what makes
            # the reported average and the paired test see the same numbers.
            for seed in range(args.seeds):
                if ("material", fold, n_mat, seed) in done:
                    continue
                rs = np.random.RandomState(1000 * seed + fold)
                sel = rs.choice(tr_mat_idx, size=n_mat, replace=False)
                rows_sel = np.flatnonzero(np.isin(row_mat, sel))
                tr_sub = fixed_row_budget(row_mat, rows_sel, sel,
                                          args.train_rows, rs)
                Xtr, ytr = R.build_X(tr_sub, state_X, row_mat, mat_X), y[tr_sub]

                t0 = time.time()
                tp = make_tabpfn()
                tp.fit(Xtr, ytr)
                mae_tp = mean_absolute_error(yte, np.asarray(tp.predict(Xte)))

                hg = HistGradientBoostingRegressor(
                    random_state=0, loss="absolute_error", max_iter=500,
                    learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=40,
                    l2_regularization=1.0, early_stopping=True,
                    n_iter_no_change=25, validation_fraction=0.1)
                hg.fit(Xtr, ytr)
                mae_hg = mean_absolute_error(yte, hg.predict(Xte))

                records.append({"split": "material", "fold": fold,
                                "n_materials": n_mat, "seed": seed,
                                "n_train_rows": len(tr_sub),
                                "mae_tabpfn": mae_tp, "mae_histgb": mae_hg})
                flush()
                print(f"fold {fold}  n_mat {n_mat:3d}  seed {seed}  "
                      f"rows {len(tr_sub):,}  tabpfn {mae_tp:.4f}  "
                      f"histgb {mae_hg:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # --- leakage control: split rows, not materials -------------------------
    if not args.skip_control:
        print("\n=== control: row-level split (deliberate leakage) ===", flush=True)
        rs = np.random.RandomState(0)
        all_rows = np.arange(len(y))
        for fold in range(args.folds):
            if ("row_leak", fold, len(mats), 0) in done:
                continue
            rsf = np.random.RandomState(500 + fold)
            perm = rsf.permutation(all_rows)
            cut = len(perm) // 5
            te_r, tr_r = perm[:cut], perm[cut:]
            te_sub = rsf.choice(te_r, min(len(te_r), 24100), replace=False)
            tr_sub = rsf.choice(tr_r, args.train_rows, replace=False)
            Xtr, ytr = R.build_X(tr_sub, state_X, row_mat, mat_X), y[tr_sub]
            Xte, yte = R.build_X(te_sub, state_X, row_mat, mat_X), y[te_sub]
            t0 = time.time()
            tp = make_tabpfn()
            tp.fit(Xtr, ytr)
            mae_tp = mean_absolute_error(yte, np.asarray(tp.predict(Xte)))
            records.append({"split": "row_leak", "fold": fold,
                            "n_materials": len(mats), "seed": 0,
                            "n_train_rows": len(tr_sub),
                            "mae_tabpfn": mae_tp, "mae_histgb": np.nan})
            flush()
            print(f"row-split fold {fold}: tabpfn {mae_tp:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(records)
    cur = df[df.split == "material"]
    print("\n=== Δε MAE vs number of materials (rows fixed) ===")
    piv = cur.groupby("n_materials")[["mae_tabpfn", "mae_histgb"]].agg(["mean", "std"])
    print(piv.round(4).to_string())

    if not args.skip_control:
        leak = df[df.split == "row_leak"].mae_tabpfn
        honest = cur[cur.n_materials == max(cur.n_materials)].mae_tabpfn
        print(f"\nmaterial-level split (all materials)  {honest.mean():.4f}")
        print(f"row-level split (leaked)              {leak.mean():.4f}")
        print(f"leakage flatters the score by         {honest.mean()-leak.mean():.4f} eV")

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
