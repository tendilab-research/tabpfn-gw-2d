"""Build the state-level QP-correction dataset for the main experiment.

For every band-matched (GW, HSE) material, each row is one electronic state
(band n, k-point) on the Gamma-M-K-Gamma path:

    target  delta = eps_GW(n,k) - eps_HSE(n,k)      [eV]
    state features: eps_HSE (vacuum-ref), eps - efermi_HSE, band index
    relative to the band edge, occupied flag, k-point (fractional + |k|),
    position along the path.

Material-level features are NOT stored here; they are joined from
gap_gw_features.csv at training time via uid.

Output: data/delta_eps_dataset.npz (float32 arrays + uid string array).
"""

import sys
from pathlib import Path

import paths

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from path_interp import load_material, match_bands

MATERIALS_ROOT = paths.MATERIALS_ROOT
OUT = paths.DATA / "delta_eps_dataset.npz"


def main():
    folders = sorted(p.parent for p in MATERIALS_ROOT.rglob("results-asr.gw.json")
                     if (p.parent / "hse.json").exists())
    cols = {k: [] for k in ["uid", "eps_hse", "eps_rel_ef", "band_rel", "occupied",
                            "kx", "ky", "k_norm", "path_frac", "delta"]}
    skipped = []
    for folder in folders:
        m = load_material(folder)
        res = match_bands(m["hse_path"], m["gw_path"], m["efermi_hse"], m["efermi_gw"])
        if res is None:
            skipped.append((m["uid"], "no constant-occupation match"))
            continue
        offset, n_occ = res
        gw, hse = m["gw_path"], m["hse_path"]
        n_gw, n_k = gw.shape
        if offset < 0 or offset + n_gw > hse.shape[0]:
            skipped.append((m["uid"], f"GW window outside HSE bands (offset={offset})"))
            continue

        hse_win = hse[offset:offset + n_gw]           # (n_gw, n_k)
        kpts = m["path_kpts"]                          # (n_k, 3) fractional
        cell2d = m["cell"][:2, :2]                     # 2D material: z row is zero
        recip2d = 2 * np.pi * np.linalg.inv(cell2d).T
        k_norm = np.linalg.norm(kpts[:, :2] @ recip2d, axis=1)

        band_idx = np.arange(n_gw)
        bands = np.repeat(band_idx, n_k)               # state grid, band-major
        cols["uid"].extend([m["uid"]] * n_gw * n_k)
        cols["eps_hse"].append(hse_win.ravel())
        cols["eps_rel_ef"].append((hse_win - m["efermi_hse"]).ravel())
        cols["band_rel"].append(bands - n_occ)         # -1 = highest valence
        cols["occupied"].append((bands < n_occ).astype(np.float32))
        cols["kx"].append(np.tile(kpts[:, 0], n_gw))
        cols["ky"].append(np.tile(kpts[:, 1], n_gw))
        cols["k_norm"].append(np.tile(k_norm, n_gw))
        cols["path_frac"].append(np.tile(np.linspace(0, 1, n_k), n_gw))
        cols["delta"].append((gw - hse_win).ravel())

    arrays = {"uid": np.array(cols.pop("uid"))}
    for k, v in cols.items():
        arrays[k] = np.concatenate([np.asarray(a, dtype=np.float32).ravel() for a in v])
    np.savez_compressed(OUT, **arrays)

    n = len(arrays["delta"])
    n_mat = len(set(arrays["uid"].tolist()))
    d = arrays["delta"]
    print(f"{n:,} states from {n_mat} materials -> {OUT}")
    print(f"delta: mean {d.mean():+.3f}, std {d.std():.3f}, "
          f"p1 {np.percentile(d,1):+.3f}, p99 {np.percentile(d,99):+.3f} eV")
    print(f"skipped {len(skipped)}: {skipped}")


if __name__ == "__main__":
    main()
