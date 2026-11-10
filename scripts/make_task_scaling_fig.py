"""Figure and statistics for the task-difficulty scaling comparison.

Reads data/task_scaling.csv (and data/material_curve.csv if present) and reports
the log-log exponent per task, with a jackknife over folds so the exponent comes
with an honest range rather than a single fit. Writes figures/fig9_task_scaling.

    python scripts/make_task_scaling_fig.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import FIGS, panel_label  # noqa: E402
import run_delta_regression as R  # noqa: E402

DATA = R.DATA
TP, HG = "#1f4e79", "#b75a22"
LABEL = {"gap_hse": r"PBE $\to$ HSE  (2,968 labels)",
         "gap_gw":  r"PBE $\to$ $G_0W_0$  (319 labels)"}


def fold_means(df, task):
    t = df[df.task == task]
    return t.groupby(["fold", "n_materials"])[["mae_tabpfn", "mae_histgb"]].mean()


def powerlaw(ns, ys):
    b, a = np.polyfit(np.log(ns), np.log(ys), 1)
    pred = np.exp(a) * np.asarray(ns, float) ** b
    r2 = 1 - ((np.log(ys) - np.log(pred)) ** 2).sum() / \
             ((np.log(ys) - np.log(ys).mean()) ** 2).sum()
    return b, np.exp(a), r2


def jackknife_exponent(piv, col):
    """Exponent range when each fold is left out in turn."""
    folds = piv.index.get_level_values("fold").unique()
    out = []
    for drop in folds:
        sub = piv[piv.index.get_level_values("fold") != drop]
        g = sub.groupby("n_materials")[col].mean()
        out.append(powerlaw(g.index.values.astype(float), g.values)[0])
    return min(out), max(out)


def main():
    df = pd.read_csv(DATA / "task_scaling.csv")
    tasks = [t for t in ("gap_hse", "gap_gw") if t in set(df.task)]

    print("=== log-log 지수 (fold 균등 평균, 잭나이프 범위) ===")
    stats = {}
    for task in tasks:
        piv = fold_means(df, task)
        g = piv.groupby("n_materials").mean()
        ns = g.index.values.astype(float)
        row = {}
        for col, name in (("mae_tabpfn", "TabPFN"), ("mae_histgb", "HistGB")):
            b, A, r2 = powerlaw(ns, g[col].values)
            lo, hi = jackknife_exponent(piv, col)
            row[col] = (b, A, r2, lo, hi)
            print(f"  {task:8s} {name:7s} exponent {b:+.3f} "
                  f"[jackknife {lo:+.3f}, {hi:+.3f}]  log-log R2 {r2:.3f}")
        stats[task] = (g, ns, row)

    if len(tasks) == 2:
        d = stats["gap_gw"][2]["mae_tabpfn"][0] - stats["gap_hse"][2]["mae_tabpfn"][0]
        lo1, hi1 = stats["gap_hse"][2]["mae_tabpfn"][3:5]
        lo2, hi2 = stats["gap_gw"][2]["mae_tabpfn"][3:5]
        overlap = not (hi1 < lo2 or hi2 < lo1)
        print(f"\n  두 과제의 TabPFN 지수 차이: {d:+.3f}"
              f"   잭나이프 범위 겹침: {'예 (차이 단정 불가)' if overlap else '아니오'}")

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    ax = axes[0]
    for task, mk in zip(tasks, ("o", "s")):
        g, ns, row = stats[task]
        ax.plot(ns, g.mae_tabpfn, mk + "-", ms=3.4, lw=1.3, color=TP,
                label=LABEL[task], mfc="none" if mk == "s" else TP)
        b, A, r2, *_ = row["mae_tabpfn"]
        ax.plot(ns, A * ns ** b, ":", lw=.9, color=TP, alpha=.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training materials")
    ax.set_ylabel("gap MAE (eV)")
    ax.set_title("TabPFN, same features, different target", pad=4)
    ax.legend(frameon=False, fontsize=5.6, loc="lower left")
    ax.grid(alpha=.25, lw=.5, which="both")
    panel_label(ax, "a")

    ax = axes[1]
    task = "gap_hse" if "gap_hse" in stats else tasks[0]
    g, ns, row = stats[task]
    for col, color, name in (("mae_tabpfn", TP, "TabPFN"),
                             ("mae_histgb", HG, "HistGB (tuned)")):
        ax.plot(ns, g[col], "o-", ms=3.4, lw=1.3, color=color, label=name)
        b, A, r2, *_ = row[col]
        ax.plot(ns, A * ns ** b, ":", lw=.9, color=color, alpha=.7)
        ax.annotate(f"$n^{{{b:.2f}}}$", xy=(ns[-1], g[col].values[-1]),
                    xytext=(-2, -9 if col == "mae_tabpfn" else 6),
                    textcoords="offset points", fontsize=6, color=color,
                    ha="right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training materials")
    ax.set_ylabel("gap MAE (eV)")
    ax.set_title(f"{task}: who follows a power law", pad=4)
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    ax.grid(alpha=.25, lw=.5, which="both")
    panel_label(ax, "b")

    fig.tight_layout()
    for ext, kw in (("png", dict(dpi=300)), ("pdf", {})):
        fig.savefig(FIGS / f"fig9_task_scaling.{ext}", **kw)
    print(f"\nsaved {FIGS/'fig9_task_scaling.png'}")


if __name__ == "__main__":
    main()
