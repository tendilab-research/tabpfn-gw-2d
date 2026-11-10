"""Build HSE DOS for all C2DB materials that have hse.json.

hse.json contains SOC eigenvalues eig_soc_mk (bands x kpts) on a uniform
Monkhorst-Pack grid (bzkpts), so a proper BZ-integrated DOS is obtained by
equal-weight Gaussian broadening:

    DOS(E) = (1/N_k) * sum_{m,k} G_sigma(E - eps_mk)

Energy convention: hse.json eigenvalues are on the raw calculator scale;
data.json scalars (vbm_hse, efermi_hse, ...) are vacuum-referenced.
We shift eigenvalues by -evac so everything is vacuum-referenced.

SOC spinor bands are singly occupied -> no spin degeneracy factor.

Output: one .npz per material in data/hse_dos/ plus manifest.csv.
The energy grid is anchored to a global lattice of multiples of STEP on the
vacuum-referenced scale, so all materials share commensurate grids.
DOS is set to NaN outside the band-coverage window (finite number of bands).
"""

import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import paths

import numpy as np

MATERIALS_ROOT = paths.MATERIALS_ROOT
OUT_DIR = paths.DATA / "hse_dos"

STEP = 0.025          # eV, grid spacing
SIGMA = 0.05          # eV, Gaussian broadening
WINDOW = (-15.0, 10.0)  # eV relative to Fermi level
COVER_MARGIN = 3 * SIGMA  # trim band-coverage window by 3 sigma


def gaussian_kernel(sigma, step):
    half = int(np.ceil(4 * sigma / step))
    x = np.arange(-half, half + 1) * step
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


KERNEL = gaussian_kernel(SIGMA, STEP)


def process_material(hse_path_str):
    hse_path = Path(hse_path_str)
    folder = hse_path.parent
    try:
        hse = json.loads(hse_path.read_text())
        data = json.loads((folder / "data.json").read_text())

        uid = data.get("uid") or folder.parent.name + "-" + folder.name
        # uid alone is unsafe on case-insensitive filesystems (2BiIn vs 2BiIN)
        file_key = str(folder.relative_to(MATERIALS_ROOT)).replace("/", "__")
        eig_raw = np.asarray(hse["eig_soc_mk"], dtype=np.float64)  # (bands, kpts)
        n_bands, n_kpts = eig_raw.shape
        efermi_raw = float(hse["efermi"])

        evac = data.get("evac")
        has_evac = evac is not None
        shift = float(evac) if has_evac else 0.0
        eig = eig_raw - shift          # vacuum-referenced (or raw if no evac)
        efermi = efermi_raw - shift

        # band-coverage window: DOS is complete only where every k-point
        # contributes bands, i.e. [max_k min_m, min_k max_m]
        cover_lo = float(eig.min(axis=0).max())
        cover_hi = float(eig.max(axis=0).min())

        # energy grid anchored to global lattice of STEP multiples
        lo = np.floor((efermi + WINDOW[0]) / STEP) * STEP
        n_pts = int(round((WINDOW[1] - WINDOW[0]) / STEP)) + 1
        energies = lo + STEP * np.arange(n_pts)

        edges = np.concatenate([energies - STEP / 2, [energies[-1] + STEP / 2]])
        counts, _ = np.histogram(eig.ravel(), bins=edges)
        dos = counts.astype(np.float64) / (n_kpts * STEP)   # states / eV / cell
        dos = np.convolve(dos, KERNEL, mode="same")

        # mask energies where the finite band set does not cover the spectrum
        valid = (energies >= cover_lo + COVER_MARGIN) & (energies <= cover_hi - COVER_MARGIN)
        dos_masked = np.where(valid, dos, np.nan)

        # validation for gapped materials: gap edges from eigenvalues
        occ = eig[eig <= efermi]
        emp = eig[eig > efermi]
        vbm_calc = float(occ.max()) if occ.size else np.nan
        cbm_calc = float(emp.min()) if emp.size else np.nan

        np.savez_compressed(
            OUT_DIR / f"{file_key}.npz",
            energies=energies.astype(np.float32),
            dos=dos_masked.astype(np.float32),
            efermi=efermi,
            evac=float(evac) if has_evac else np.nan,
            sigma=SIGMA,
            cover_lo=cover_lo,
            cover_hi=cover_hi,
        )

        return {
            "uid": uid,
            "file": f"{file_key}.npz",
            "folder": str(folder.relative_to(MATERIALS_ROOT)),
            "n_bands": n_bands,
            "n_kpts": n_kpts,
            "has_evac": has_evac,
            "efermi": round(efermi, 6),
            "cover_lo": round(cover_lo, 3),
            "cover_hi": round(cover_hi, 3),
            "vbm_calc": round(vbm_calc, 6) if np.isfinite(vbm_calc) else "",
            "cbm_calc": round(cbm_calc, 6) if np.isfinite(cbm_calc) else "",
            "vbm_hse": data.get("vbm_hse"),
            "cbm_hse": data.get("cbm_hse"),
            "gap_hse": data.get("gap_hse"),
            "error": "",
        }
    except Exception as exc:  # keep batch running, record failure
        return {"uid": "", "folder": str(folder), "error": repr(exc)}


def main():
    paths = sorted(str(p) for p in MATERIALS_ROOT.rglob("hse.json"))
    if len(sys.argv) > 1:  # optional limit for smoke tests
        paths = paths[: int(sys.argv[1])]
    print(f"processing {len(paths)} materials", flush=True)

    with Pool(8) as pool:
        rows = []
        for i, row in enumerate(pool.imap_unordered(process_material, paths, chunksize=16)):
            rows.append(row)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(paths)}", flush=True)

    fields = ["uid", "file", "folder", "n_bands", "n_kpts", "has_evac", "efermi",
              "cover_lo", "cover_hi", "vbm_calc", "cbm_calc",
              "vbm_hse", "cbm_hse", "gap_hse", "error"]
    with open(OUT_DIR.parent / "hse_dos_manifest.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    errors = [r for r in rows if r["error"]]
    print(f"done: {len(rows) - len(errors)} ok, {len(errors)} failed")
    for r in errors[:10]:
        print("  FAIL", r["folder"], r["error"])


if __name__ == "__main__":
    main()
