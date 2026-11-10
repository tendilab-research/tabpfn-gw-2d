"""Interpolate HSE eigenvalues from the uniform BZ grid onto the GW band path.

hse.json holds eig_soc_mk on a Gamma-centered uniform grid (bzkpts, fractional
coords). results-asr.gw.json holds GW quasiparticle eigenvalues e_int_mk on a
high-symmetry path (path.kpts, fractional). Periodic bilinear interpolation of
each (sorted) HSE band onto the path k-points makes the two directly
comparable state-by-state - same measure, same path.

This mirrors what C2DB's own asr.hse recipe does to build its published HSE
band structures (monkhorst_pack_interpolate on the same grid data).
"""

import json
from pathlib import Path

import numpy as np


def load_json(path):
    return json.loads(Path(path).read_text())


def ndarray(obj):
    """Decode ASE/ASR {'__ndarray__': [shape, dtype, flatdata]} JSON blobs.
    ASE Cell objects wrap the ndarray one level deeper ({'array': {...}})."""
    if "array" in obj and "__ndarray__" not in obj:
        obj = obj["array"]
    shape, dtype, data = obj["__ndarray__"]
    return np.asarray(data, dtype=dtype).reshape(shape)


def grid_from_bzkpts(bzkpts):
    """Map fractional k-points of a uniform Gamma-centered grid to integer
    indices. Returns (Nx, Ny, index_array) where index_array[k] = (i, j)."""
    k = np.asarray(bzkpts, dtype=np.float64)
    assert np.allclose(k[:, 2], 0.0), "expected 2D material (kz == 0)"
    kf = np.mod(k[:, :2], 1.0)
    n = len(k)
    for nx in range(1, n + 1):
        if n % nx:
            continue
        ny = n // nx
        idx = np.stack([np.round(kf[:, 0] * nx), np.round(kf[:, 1] * ny)], axis=1)
        if not (np.allclose(kf[:, 0] * nx, idx[:, 0], atol=1e-6)
                and np.allclose(kf[:, 1] * ny, idx[:, 1], atol=1e-6)):
            continue
        ij = (idx[:, 0].astype(int) % nx) * ny + (idx[:, 1].astype(int) % ny)
        if len(set(ij.tolist())) == n:  # every grid point hit exactly once
            return nx, ny, idx.astype(int)
    raise ValueError("bzkpts do not form a uniform Gamma-centered grid")


def interpolate_to_path(eig_mk, bzkpts, path_kpts):
    """Periodic bilinear interpolation of eig_mk (bands x kpts on uniform
    grid) onto path_kpts (fractional). Returns (bands x n_path)."""
    eig_mk = np.asarray(eig_mk, dtype=np.float64)
    nx, ny, idx = grid_from_bzkpts(bzkpts)
    nb = eig_mk.shape[0]

    grid = np.empty((nb, nx, ny))
    grid[:, idx[:, 0] % nx, idx[:, 1] % ny] = eig_mk

    kp = np.asarray(path_kpts, dtype=np.float64)
    tx = np.mod(kp[:, 0], 1.0) * nx
    ty = np.mod(kp[:, 1], 1.0) * ny
    i0 = np.floor(tx).astype(int) % nx
    j0 = np.floor(ty).astype(int) % ny
    i1, j1 = (i0 + 1) % nx, (j0 + 1) % ny
    fx, fy = tx - np.floor(tx), ty - np.floor(ty)

    return (grid[:, i0, j0] * (1 - fx) * (1 - fy)
            + grid[:, i1, j0] * fx * (1 - fy)
            + grid[:, i0, j1] * (1 - fx) * fy
            + grid[:, i1, j1] * fx * fy)


def load_material(folder):
    """Load and align everything needed for HSE/GW path comparison.

    Returns dict with vacuum-referenced arrays:
      hse_path (n_hse_bands x n_path), gw_path (n_gw_bands x n_path),
      efermi_hse, efermi_gw, path_kpts, plus data.json scalars.
    """
    folder = Path(folder)
    data = load_json(folder / "data.json")
    hse = load_json(folder / "hse.json")
    gw = load_json(folder / "results-asr.gw.json")
    if "kwargs" in gw:  # newer ASR wraps results; older files are flat
        gw = gw["kwargs"]["data"]

    evac = float(data["evac"])
    bs = gw["bandstructure"]
    path_kpts = ndarray(bs["path"]["kpts"])
    gw_path = ndarray(bs["e_int_mk"]) - evac          # SOC, bands x kpts

    eig_mk = np.asarray(hse["eig_soc_mk"], dtype=np.float64) - evac
    hse_path = interpolate_to_path(eig_mk, hse["bzkpts"], path_kpts)

    return {
        "uid": data.get("uid"),
        "data": data,
        "cell": ndarray(bs["path"]["cell"]),
        "path_kpts": path_kpts,
        "hse_path": hse_path,
        "gw_path": gw_path,
        "efermi_hse": float(hse["efermi"]) - evac,
        "efermi_gw": float(gw["efermi_gw_soc"]) - evac,
    }


def match_bands(hse_path, gw_path, efermi_hse, efermi_gw):
    """Align the GW band window inside the full HSE band set by counting
    occupied bands at each k. Returns (offset, n_occ_gw) or None if the
    occupied count is not constant along the path (metallic / mismatch)."""
    occ_gw = (gw_path <= efermi_gw).sum(axis=0)
    occ_hse = (hse_path <= efermi_hse).sum(axis=0)
    if len(set(occ_gw.tolist())) != 1 or len(set(occ_hse.tolist())) != 1:
        return None
    return int(occ_hse[0]) - int(occ_gw[0]), int(occ_gw[0])
