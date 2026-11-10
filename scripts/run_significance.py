"""Every significance test the paper's claims rest on, in one place.

With five or ten cross-validation folds a paired t-test is easy to over-read, so
each claim is reported four ways and the weakest one governs:

  paired t          the conventional number
  exact sign-flip   the non-parametric floor. With n folds only 2^n sign
                    assignments exist, so the smallest attainable two-sided p
                    is 2/2^n - 0.0625 at n=5, 0.002 at n=10. A t-test p below
                    that floor is produced by the normality assumption, not by
                    the data.
  Holm              the paper quotes several interval tests side by side
  jackknife         leave-one-fold-out. At this n it is more honest than any p:
                    a claim that dies when one fold is removed is not a result.

Reads whichever result files are present and skips the rest. Writes
data/significance.csv.

    python scripts/run_significance.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

try:
    from scipy.stats import ttest_rel
except ImportError:                                        # scipy is optional
    def ttest_rel(a, b):
        d = np.asarray(a, float) - np.asarray(b, float)
        n = len(d)
        t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
        from math import erf, sqrt
        z = abs(t) * np.sqrt(n / (n + t ** 2))             # crude but bounded
        p = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
        return type("R", (), {"pvalue": float(min(1.0, p))})


def exact_signflip_p(diff):
    """Two-sided p over all 2^n sign assignments. Exact, assumption-free."""
    d = np.asarray(diff, float)
    n = len(d)
    if n > 22:
        return np.nan
    obs = abs(d.mean())
    hits = sum(1 for s in itertools.product([1, -1], repeat=n)
               if abs(np.mean(d * np.asarray(s))) >= obs - 1e-15)
    return hits / 2 ** n


def jackknife(a, b):
    """Range of paired-t p when each fold is left out in turn."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ps = [ttest_rel(np.delete(a, i), np.delete(b, i)).pvalue for i in range(len(a))]
    return min(ps), max(ps), sum(p > 0.05 for p in ps), len(ps)


def cohen_dz(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return d.mean() / d.std(ddof=1)


def holm(pvals):
    order = np.argsort(pvals)
    m, out, prev = len(pvals), np.empty(len(pvals)), 0.0
    for rank, idx in enumerate(order):
        prev = max(prev, min(1.0, (m - rank) * pvals[idx]))
        out[idx] = prev
    return out


def test(name, a, b, family=None, better="b"):
    """`a` and `b` are per-fold errors; `better` names the side claimed lower."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    p = ttest_rel(a, b).pvalue
    lo, hi, n_bad, n = jackknife(a, b)
    ex = exact_signflip_p(d)
    return {"claim": name, "family": family or "", "n_folds": n,
            "mean_diff_eV": d.mean(), "wins": int((d > 0).sum()) if better == "b"
            else int((d < 0).sum()),
            "paired_t_p": p, "exact_signflip_p": ex,
            "exact_floor": 2 / 2 ** n, "cohen_dz": cohen_dz(a, b),
            "jackknife_p_min": lo, "jackknife_p_max": hi,
            "jackknife_fails": n_bad,
            "verdict": ("fragile" if n_bad > 0 else
                        "robust" if p < 0.05 else "not significant")}


def fold_means(df, value, by, task=None):
    if task is not None:
        df = df[df.task == task]
    return df.groupby(["fold", by])[value].mean().unstack()


def main():
    rows = []
    D = paths.DATA

    # --- baseline: does a 185-fold row increase help? ----------------------
    f = D / "baseline_tuning.csv"
    if f.exists():
        b = pd.read_csv(f)
        fc = [c for c in b.columns if c.startswith("fold")]
        if fc:
            for cfg in b.config.unique():
                sub = b[(b.config == cfg) & (b.train_scope == "sub")][fc].to_numpy()[0]
                ful = b[(b.config == cfg) & (b.train_scope == "full")][fc].to_numpy()[0]
                rows.append(test(f"rows: 1e4 vs 1.9e6 ({cfg})", sub, ful,
                                 family="row budget"))

    # --- material-count curve ---------------------------------------------
    for tag, fn in (("5-fold", "material_curve.csv"),
                    ("10-fold", "material_curve_f10.csv")):
        f = D / fn
        if not f.exists():
            continue
        m = pd.read_csv(f)
        m = m[m.split == "material"]
        piv = fold_means(m, "mae_tabpfn", "n_materials")
        cols = [c for c in piv.columns if piv[c].notna().sum() >= 3]
        for a, b_ in zip(cols[:-1], cols[1:]):
            x, y = piv[a].dropna(), piv[b_].dropna()
            i = x.index.intersection(y.index)
            if len(i) >= 3:
                rows.append(test(f"materials {a}->{b_} ({tag})",
                                 x[i], y[i], family=f"curve {tag}"))
        # leakage control
        leak = pd.read_csv(f)
        leak = leak[leak.split == "row_leak"]
        if len(leak) >= 3:
            honest = piv[cols[-1]].dropna()
            rows.append({"claim": f"row-split leakage ({tag})", "family": "leakage",
                         "n_folds": len(leak), "mean_diff_eV":
                             honest.mean() - leak.mae_tabpfn.mean(),
                         "wins": len(leak), "paired_t_p": np.nan,
                         "exact_signflip_p": np.nan, "exact_floor": np.nan,
                         "cohen_dz": np.nan, "jackknife_p_min": np.nan,
                         "jackknife_p_max": np.nan, "jackknife_fails": 0,
                         "verdict": f"understates by "
                                    f"{100*(1-leak.mae_tabpfn.mean()/honest.mean()):.0f}%"})

    # --- task-difficulty scaling: model gap at the largest count -----------
    f = D / "task_scaling.csv"
    if f.exists():
        t = pd.read_csv(f)
        for task in t.task.unique():
            tp = fold_means(t, "mae_tabpfn", "n_materials", task)
            hg = fold_means(t, "mae_histgb", "n_materials", task)
            big = max(tp.columns)
            rows.append(test(f"TabPFN vs HistGB at n={big} ({task})",
                             hg[big], tp[big], family="model gap"))
            cols = list(tp.columns)
            rows.append(test(f"materials {cols[-2]}->{cols[-1]} ({task})",
                             tp[cols[-2]], tp[cols[-1]], family=f"curve {task}"))

    if not rows:
        raise SystemExit("no result files found in " + str(D))

    out = pd.DataFrame(rows)

    # Holm within each family of simultaneously quoted tests
    out["holm_p"] = np.nan
    for fam, g in out.groupby("family"):
        ok = g.paired_t_p.notna()
        if fam and ok.sum() > 1:
            out.loc[g.index[ok], "holm_p"] = holm(g.paired_t_p[ok].to_numpy())

    out.to_csv(D / "significance.csv", index=False)

    pd.set_option("display.width", 200)
    show = out[["claim", "n_folds", "mean_diff_eV", "wins", "paired_t_p",
                "exact_signflip_p", "exact_floor", "holm_p",
                "jackknife_fails", "verdict"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote {D/'significance.csv'}")
    print("\nRead the exact_floor column first: a paired-t p below it is an "
          "artefact of the normality assumption, not evidence.")


if __name__ == "__main__":
    main()
