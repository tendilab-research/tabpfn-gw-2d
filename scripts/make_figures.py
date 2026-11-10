"""Nature Communications figure set (figures/, PNG 300dpi + PDF).

fig1_pipeline           : concept/pipeline schematic
fig2_gap_regression     : OOF scatter, non-magnetic scope (319 materials)
fig3_delta_benchmark    : (a) model benchmark bars, (b) per-material error
                          distribution (final non-magnetic model),
                          (c) error vs gap with magnetic overlay (scope just.)
fig4_dos_reconstruction : held-out DOS reconstruction (3 materials)

Style follows Nature Portfolio conventions: bold lowercase panel labels,
Helvetica, thin spines, muted palette.
"""

from pathlib import Path

import paths

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DATA = paths.DATA
FIGS = paths.FIGS

BLUE, ORANGE, PURPLE = "#6F8AC9", "#F0A860", "#A678B4"
RED, GRAY, DARK = "#C96A6A", "#B9B9B9", "#3A3A3A"
LBLUE, LORANGE, LPURPLE, LGREEN = "#E3E9F7", "#FBEDDC", "#EFE4F3", "#E5F0DF"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "legend.fontsize": 6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "figure.dpi": 110, "savefig.bbox": "tight",
})


def panel_label(ax, letter, dx=-0.18, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


def save(fig, name, dpi=300):
    fig.savefig(FIGS / f"{name}.png", dpi=dpi)
    fig.savefig(FIGS / f"{name}.pdf")
    plt.close(fig)
    print(f"saved {name}")


def fig1_pipeline():
    fig = plt.figure(figsize=(7.2, 2.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.72, 1.0], wspace=0.16,
                          left=0.005, right=0.985, top=0.88, bottom=0.13)

    # -- panel a: workflow -------------------------------------------------
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(-0.08, 10.14)
    ax.set_ylim(0.25, 3.05)
    ax.axis("off")

    STAGE = [  # (accent, tint)
        ("#5B84C4", "#EDF2FA"), ("#E8985A", "#FDF1E6"),
        ("#9A7BB5", "#F3EDF8"), ("#6BAB79", "#EAF4ED"),
    ]

    def box(x, w, stage, title, lines, title_size=7.0):
        color, tint = STAGE[stage]
        ax.add_patch(FancyBboxPatch(
            (x, 0.42), w, 1.80, boxstyle="round,pad=0.05,rounding_size=0.12",
            fc=tint, ec=color, lw=1.1))
        ax.add_patch(FancyBboxPatch(
            (x + 0.08, 1.80), w - 0.16, 0.34,
            boxstyle="round,pad=0.045,rounding_size=0.09", fc=color, ec="none"))
        ax.text(x + w / 2, 1.97, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color="white")
        ax.text(x + w / 2, 1.06, lines, ha="center", va="center",
                fontsize=6.1, color=DARK, linespacing=1.9)

    box(0.05, 1.95, 0, "C2DB 2D crystals",
        "16,905 structures\n339 with $G_0W_0$\n241 non-magnetic")
    box(2.50, 2.60, 1, "One row per state",
        "$\\varepsilon_{HSE}(n,\\mathbf{k})$ · band · $\\mathbf{k}$\n"
        "composition · DFT scalars\n$2.5\\times10^6$ states")
    box(5.60, 2.00, 2, "TabPFN",
        "prior-fitted\ntransformer\nin-context learning")
    box(8.10, 1.90, 3, "$G_0W_0$ spectra", title_size=6.8,
        lines="$\\varepsilon_{HSE} + \\Delta\\varepsilon(n,\\mathbf{k})$\n"
        "DOS · gap\nband edges")

    for (x0, x1), lab in [
            ((2.06, 2.44), "vacuum align\npath interp.\nband match"),
            ((5.16, 5.54), "$10^4$ rows\nmaterial-level\nfolds"),
            ((7.66, 8.04), "state-wise $\\Delta\\varepsilon$\n+ 90% CI")]:
        ax.add_patch(FancyArrowPatch((x0, 1.32), (x1, 1.32),
                                     arrowstyle="-|>", mutation_scale=10,
                                     color=DARK, lw=1.0))
        ax.text((x0 + x1) / 2, 2.50, lab, ha="center", va="center",
                fontsize=5.6, color="#7A7A7A", linespacing=1.5, style="italic")
    ax.text(-0.005, 1.02, "a", transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left")

    # -- panel b: state-resolved Delta-eps on MoS2 -------------------------
    axb = fig.add_subplot(gs[1])
    d = np.load(DATA / "delta_eps_dataset.npz", allow_pickle=True)
    m = d["uid"] == "1MoS2-1"
    pf, eps, dlt = d["path_frac"][m], d["eps_hse"][m], d["delta"][m]
    occ, band = d["occupied"][m], d["band_rel"][m]
    kx, ky = d["kx"][m], d["ky"][m]

    vbm = eps[occ == 1].max()
    cbm_gw = (eps + dlt)[occ == 0].min()
    lo, hi = vbm - 2.1, cbm_gw + 2.1

    for b in np.unique(band):
        sel = band == b
        o = np.argsort(pf[sel])
        if eps[sel].min() > hi or (eps[sel] + dlt[sel]).max() < lo:
            continue
        axb.plot(pf[sel][o], eps[sel][o], color=GRAY, lw=0.7, zorder=1)
        axb.plot(pf[sel][o], (eps[sel] + dlt[sel])[o], color="#5B84C4",
                 lw=0.9, zorder=2)

    # high-symmetry ticks from reduced k-points
    def pf_at(target):
        dd = np.stack([kx, ky], 1) - target
        dd = np.abs((dd + 0.5) % 1.0 - 0.5)
        return pf[np.argmin(np.hypot(dd[:, 0], dd[:, 1]))]

    tM, tK = pf_at(np.array([0.5, 0.0])), pf_at(np.array([1 / 3, 1 / 3]))
    for t in (tM, tK):
        axb.axvline(t, color="#DDDDDD", lw=0.5, zorder=0)
    axb.set_xticks([0, tM, tK, 1],
                   ["$\\Gamma$", "M", "K", "$\\Gamma$"])

    # annotate the QP correction at the CBM
    iu = np.flatnonzero(occ == 0)
    i0 = iu[np.argmin(eps[iu])]
    axb.add_patch(FancyArrowPatch(
        (pf[i0], eps[i0]), (pf[i0], eps[i0] + dlt[i0]),
        arrowstyle="-|>", mutation_scale=7, color=RED, lw=1.1, zorder=3))
    axb.text(pf[i0], eps[i0] - 0.35,
             "$\\Delta\\varepsilon(n,\\mathbf{k})$", fontsize=6.8,
             color=RED, va="top", ha="center",
             bbox=dict(boxstyle="round,pad=0.18", fc="white",
                       ec="none", alpha=0.9))

    axb.plot([], [], color=GRAY, lw=1.0, label="HSE (input)")
    axb.plot([], [], color="#5B84C4", lw=1.0, label="$G_0W_0$ (target)")
    axb.legend(loc="center left", bbox_to_anchor=(0.03, 0.46),
               fontsize=6.2, handlelength=1.3, labelspacing=0.4)
    axb.set_xlim(0, 1)
    axb.set_ylim(lo, hi)
    axb.set_ylabel("$E - E_{vac}$ (eV)")
    axb.set_title("MoS$_2$", fontsize=7.5, pad=8)
    panel_label(axb, "b", dx=-0.24)
    save(fig, "fig1_pipeline", dpi=600)


def fig2_gap_regression():
    oof_path = DATA / "gap_gw_oof_nomag.npz"
    res_path = DATA / "gap_gw_cv_results_nomag.csv"
    if not (oof_path.exists() and res_path.exists()):
        print("fig2: inputs not ready, skipped")
        return
    z = np.load(oof_path, allow_pickle=True)
    res = pd.read_csv(res_path, index_col=0)
    y = z["y"]
    panels = [("lin_gap_hse", "Linear ($E_g^{HSE}$)"),
              ("rf_all", "Random forest"),
              ("tabpfn_all", "TabPFN")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharex=True, sharey=True)
    lim = [0, 12]
    for i, (ax, (key, title)) in enumerate(zip(axes, panels)):
        p, mae = z[f"pred_{key}"], res.loc[key, "mae"]
        ax.plot(lim, lim, ls="--", lw=0.6, color=GRAY, zorder=1)
        ax.scatter(y, p, s=6, alpha=0.55, color=BLUE, edgecolors="none", zorder=2)
        ax.text(0.05, 0.92, f"MAE = {mae:.3f} eV", transform=ax.transAxes,
                fontsize=6.5, va="top")
        ax.set_title(title, pad=4)
        ax.set_xlabel("$E_g^{G_0W_0}$, calculated (eV)")
        ax.set(xlim=lim, ylim=lim, aspect="equal")
        panel_label(ax, chr(97 + i), dx=-0.22 if i == 0 else -0.12)
    axes[0].set_ylabel("$E_g^{G_0W_0}$, predicted (eV)")
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig2_gap_regression")


def fig3_delta_benchmark():
    res_path = DATA / "delta_cv_results_cloud_nomag.csv"
    pm_path = DATA / "delta_per_material_mae_nomag.csv"
    pm_mixed_path = DATA / "delta_per_material_mae.csv"
    if not (res_path.exists() and pm_path.exists()):
        print("fig3: inputs not ready, skipped")
        return
    res = pd.read_csv(res_path).set_index("model")
    order = ["zero", "scissors", "linear_sub", "histgb_sub", "histgb_full", "tabpfn"]
    labels = ["HSE\n(no corr.)", "scissors", "linear",
              "GBDT\n($10^4$)", "GBDT\n($2{\\times}10^6$)", "TabPFN\n($10^4$)"]
    colors = [GRAY, GRAY, GRAY, ORANGE, ORANGE, BLUE]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4),
                             gridspec_kw={"width_ratios": [1.15, 1, 1]})
    ax = axes[0]
    x = np.arange(len(order))
    ax.bar(x, res.loc[order, "mae"], yerr=res.loc[order, "mae_std"],
           color=colors, width=0.62, error_kw=dict(lw=0.7, capsize=2, capthick=0.7))
    for xi, m in zip(x, order):
        ax.text(xi, res.loc[m, "mae"] + res.loc[m, "mae_std"] + 0.012,
                f"{res.loc[m, 'mae']:.3f}", ha="center", fontsize=5.5)
    ax.set_xticks(x, labels, fontsize=5.8)
    ax.set_ylabel(r"$\Delta\varepsilon$ MAE (eV)")
    ax.set_ylim(0, 0.56)
    panel_label(ax, "a", dx=-0.20)

    pm = pd.read_csv(pm_path, index_col=0)
    ax = axes[1]
    bins = np.logspace(np.log10(0.01), np.log10(1.0), 26)
    ax.hist(pm.mae, bins=bins, color=BLUE, alpha=0.85)
    ax.axvline(pm.mae.median(), color=DARK, lw=0.8, ls="--")
    ax.text(pm.mae.median() * 1.15, ax.get_ylim()[1] * 0.93,
            f"median {pm.mae.median():.3f} eV", fontsize=6, va="top")
    ax.set_xscale("log")
    ax.set_xlabel(r"per-material $\Delta\varepsilon$ MAE (eV)")
    ax.set_ylabel("number of materials")
    panel_label(ax, "b", dx=-0.20)

    ax = axes[2]
    ax.scatter(pm.gap_gw, pm.mae, s=8, alpha=0.6, color=BLUE,
               edgecolors="none", label="non-magnetic (final model)")
    if pm_mixed_path.exists():
        pmm = pd.read_csv(pm_mixed_path, index_col=0)
        mg = pmm[pmm.magnetic]
        ax.scatter(mg.gap_gw, mg.mae, s=8, alpha=0.75, color=RED,
                   edgecolors="none", label="magnetic (mixed-scope model)")
    ax.set_yscale("log")
    ax.set_xlabel("$E_g^{G_0W_0}$ (eV)")
    ax.set_ylabel(r"per-material $\Delta\varepsilon$ MAE (eV)")
    ax.legend(handlelength=1.2, loc="upper right", fontsize=5.2)
    panel_label(ax, "c", dx=-0.24)
    fig.tight_layout(w_pad=1.8)
    save(fig, "fig3_delta_benchmark")


def fig4_dos_reconstruction():
    cur_path = DATA / "dos_reconstruction_curves.npz"
    sum_path = DATA / "dos_reconstruction_summary.csv"
    if not (cur_path.exists() and sum_path.exists()):
        print("fig4: inputs not ready, skipped")
        return
    z = np.load(cur_path, allow_pickle=True)
    summary = pd.read_csv(sum_path).set_index("uid")
    uids = z["uids"].tolist()
    fig, axes = plt.subplots(len(uids), 1, figsize=(7.2, 1.75 * len(uids)))
    for i, (ax, uid) in enumerate(zip(axes, uids)):
        E, d_hse = z[f"{uid}:E"], z[f"{uid}:hse"]
        d_true, d_pred = z[f"{uid}:true"], z[f"{uid}:pred"]
        ax.fill_between(E, d_hse, color=GRAY, alpha=0.45, lw=0, label="HSE06")
        ax.plot(E, d_true, color=DARK, lw=0.9, label="$G_0W_0$")
        ax.plot(E, d_pred, color=RED, lw=0.9, ls="--", label="$G_0W_0$ (TabPFN)")
        s = summary.loc[uid]
        raw = uid.lstrip("0123456789").rsplit("-", 1)[0]
        name = "".join(f"$_{c}$" if c.isdigit() else c for c in raw)
        ax.text(0.99, 0.90, name, transform=ax.transAxes, ha="right",
                va="top", fontsize=7.5, fontweight="bold")
        ax.text(0.99, 0.72,
                f"$\\Delta\\varepsilon$ MAE = {s.mae_state:.3f} eV\n"
                f"$E_g$: {s.gap_true:.2f} $\\rightarrow$ {s.gap_pred:.2f} eV",
                transform=ax.transAxes, ha="right", va="top", fontsize=6)
        ax.set_ylabel("DOS (states eV$^{-1}$)")
        ax.set_ylim(bottom=0)
        ax.margins(x=0.01)
        panel_label(ax, chr(97 + i), dx=-0.07)
        if i == 0:
            ax.legend(loc="upper left", ncol=3, handlelength=1.4)
    axes[-1].set_xlabel("$E - E_{\\mathrm{vac}}$ (eV)")
    fig.tight_layout(h_pad=1.2)
    save(fig, "fig4_dos_reconstruction")


def fig5_uq_screening():
    uq_path = DATA / "delta_uq_calibration.npz"
    scr_path = DATA / "screening_release.csv"
    if not (uq_path.exists() and scr_path.exists()):
        print("fig5: inputs not ready, skipped")
        return
    z = np.load(uq_path)
    levels, q, y = z["quantile_levels"], z["q"], z["y"]
    scr = pd.read_csv(scr_path)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4),
                             gridspec_kw={"width_ratios": [1, 1, 1.25]})
    ax = axes[0]
    emp = [(y <= q[i]).mean() for i in range(len(levels))]
    ax.plot([0, 1], [0, 1], ls="--", lw=0.6, color=GRAY)
    ax.plot(levels, emp, "o-", color=BLUE, lw=0.9, ms=3)
    ax.set_xlabel("nominal quantile level")
    ax.set_ylabel("empirical $P(y \\leq q)$")
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
    ax.text(0.05, 0.92, "90% CI coverage: 94.5%", transform=ax.transAxes,
            fontsize=6, va="top")
    panel_label(ax, "a", dx=-0.24)

    ax = axes[1]
    w90 = q[-1] - q[0]
    ax.hist(w90, bins=np.logspace(np.log10(0.02), np.log10(5), 30),
            color=BLUE, alpha=0.85)
    ax.axvline(np.median(w90), color=DARK, lw=0.8, ls="--")
    ax.text(np.median(w90) * 1.2, ax.get_ylim()[1] * 0.93,
            f"median {np.median(w90):.2f} eV", fontsize=6, va="top")
    ax.set_xscale("log")
    ax.set_xlabel(r"90% CI width for $\Delta\varepsilon$ (eV)")
    ax.set_ylabel("states")
    panel_label(ax, "b", dx=-0.22)

    ax = axes[2]
    order = np.argsort(scr.gap_hse.to_numpy())
    gh = scr.gap_hse.to_numpy()[order]
    ax.fill_between(gh, scr.gap_gw_q05.to_numpy()[order],
                    scr.gap_gw_q95.to_numpy()[order],
                    color=BLUE, alpha=0.18, lw=0, label="90% CI")
    ax.scatter(scr.gap_hse, scr.gap_gw_pred, s=3, alpha=0.4, color=BLUE,
               edgecolors="none", label="predicted $E_g^{G_0W_0}$")
    lim = [0, 13]
    ax.plot(lim, lim, ls="--", lw=0.6, color=GRAY)
    ax.set_xlabel("$E_g^{HSE}$ (eV)")
    ax.set_ylabel("predicted $E_g^{G_0W_0}$ (eV)")
    ax.set(xlim=(0, 9), ylim=(0, 13))
    ax.text(0.05, 0.92, f"{len(scr):,} materials\n(no prior GW)",
            transform=ax.transAxes, fontsize=6, va="top")
    ax.legend(loc="lower right", handlelength=1.2, fontsize=5.5)
    panel_label(ax, "c", dx=-0.20)
    fig.tight_layout(w_pad=1.8)
    save(fig, "fig5_uq_screening")


def fig6_ood_active():
    ood_path = DATA / "ood_results.csv"
    al_path = DATA / "active_learning.csv"
    if not ood_path.exists():
        print("fig6: inputs not ready, skipped")
        return
    ood = pd.read_csv(ood_path)
    have_al = al_path.exists()

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    ax = axes[0]
    splits = ["random", "loeo", "prototype"]
    slabels = ["random", "leave-one-\nelement-out", "out-of-\nprototype"]
    agg = ood.groupby("split").apply(
        lambda g: pd.Series({
            "tabpfn": np.average(g.mae_tabpfn, weights=g.n_test),
            "histgb": np.average(g.mae_histgb, weights=g.n_test)}))
    x = np.arange(len(splits))
    ax.bar(x - 0.17, agg.loc[splits, "tabpfn"], 0.34, color=BLUE, label="TabPFN")
    ax.bar(x + 0.17, agg.loc[splits, "histgb"], 0.34, color=ORANGE, label="GBDT")
    ax.set_xticks(x, slabels, fontsize=6)
    ax.set_ylabel("$E_g^{G_0W_0}$ MAE (eV)")
    ax.legend(handlelength=1.2)
    panel_label(ax, "a", dx=-0.22)

    ax = axes[1]
    cov = ood.groupby("split").apply(
        lambda g: np.average(g.coverage90, weights=g.n_test))
    wid = ood.groupby("split").apply(
        lambda g: np.average(g.ci_width_median, weights=g.n_test))
    ax.bar(x, wid.loc[splits], 0.5, color=BLUE, alpha=0.85)
    for xi, s in zip(x, splits):
        ax.text(xi, wid.loc[s] + 0.03, f"cov.\n{cov.loc[s]:.2f}",
                ha="center", fontsize=5.5)
    ax.axhline(0, color=DARK, lw=0.6)
    ax.set_xticks(x, slabels, fontsize=6)
    ax.set_ylabel("median 90% CI width (eV)")
    ax.set_ylim(0, 1.25)
    panel_label(ax, "b", dx=-0.20)

    ax = axes[2]
    if have_al:
        al = pd.read_csv(al_path)
        for strat, color, label in [("random", GRAY, "random acquisition"),
                                    ("uncertainty", BLUE, "uncertainty acquisition")]:
            g = al[al.strategy == strat].groupby("n_labels").test_mae
            mean, std = g.mean(), g.std()
            ax.plot(mean.index, mean, "o-", color=color, lw=0.9, ms=2.5,
                    label=label)
            ax.fill_between(mean.index, mean - std, mean + std,
                            color=color, alpha=0.18, lw=0)
        ax.set_xlabel("number of $G_0W_0$ labels")
        ax.set_ylabel("test MAE (eV)")
        ax.legend(handlelength=1.4)
    else:
        ax.text(0.5, 0.5, "active learning\npending", transform=ax.transAxes,
                ha="center")
    panel_label(ax, "c", dx=-0.22)
    fig.tight_layout(w_pad=1.8)
    save(fig, "fig6_ood_active")


def fig7_literature():
    lit_path = DATA / "literature_gw.csv"
    oof_path = DATA / "gap_gw_oof_nomag.npz"
    if not (lit_path.exists() and oof_path.exists()):
        print("fig7: inputs not ready, skipped")
        return
    lit = pd.read_csv(lit_path)
    z = np.load(oof_path, allow_pickle=True)
    uids = z["uid"].tolist()

    mats = ["MoS2", "WS2", "WSe2", "MoSe2", "InSe", "GaSe", "BN"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for yi, mat in enumerate(mats):
        g = lit[lit.material == mat]
        uid = g.uid.iloc[0]
        for _, r in g.iterrows():
            marker = "D" if r.soc == "yes" else "o"
            ax.scatter(r.value_eV, yi, marker=marker, s=22, facecolors="none",
                       edgecolors=DARK, linewidths=0.8, zorder=3)
        i = uids.index(uid)
        ax.scatter(z["y"][i], yi, marker="s", s=26, color=DARK, zorder=4)
        ax.scatter(z["pred_tabpfn_all"][i], yi, marker="x", s=34, color=RED,
                   linewidths=1.4, zorder=5)
    ax.scatter([], [], marker="o", facecolors="none", edgecolors=DARK,
               label="literature $G_0W_0$ (no SOC)")
    ax.scatter([], [], marker="D", facecolors="none", edgecolors=DARK,
               label="literature $G_0W_0$ (SOC)")
    ax.scatter([], [], marker="s", color=DARK, label="C2DB $G_0W_0$ (SOC)")
    ax.scatter([], [], marker="x", color=RED, label="TabPFN (out-of-fold)")
    names = ["MoS$_2$", "WS$_2$", "WSe$_2$", "MoSe$_2$", "InSe", "GaSe", "hBN"]
    ax.set_yticks(range(len(mats)), names)
    ax.set_xlabel("quasiparticle gap (eV)")
    ax.legend(loc="lower right", fontsize=5.5, handlelength=1.0)
    ax.set_xlim(1.5, 8.8)
    fig.tight_layout()
    save(fig, "fig7_literature")


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    fig1_pipeline()
    fig2_gap_regression()
    fig3_delta_benchmark()
    fig4_dos_reconstruction()
    fig5_uq_screening()
    fig6_ood_active()
    fig7_literature()
