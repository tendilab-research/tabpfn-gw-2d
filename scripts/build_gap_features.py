"""Build the tabular feature set for gap_gw regression (339 GW materials).

One row per material. Features come from data.json scalars available for all
339 materials plus simple structure-derived statistics. GW-derived keys
(gap_gw target aside) and BSE products (E_B) are excluded to avoid leakage.
"""

import json
from pathlib import Path

import paths

import numpy as np
import pandas as pd

MATERIALS_ROOT = paths.MATERIALS_ROOT
OUT = paths.DATA / "gap_gw_features.csv"

DATA_KEYS = [
    # PBE electronic structure
    "gap", "gap_dir", "gap_dir_nosoc", "vbm", "cbm", "efermi",
    # HSE electronic structure
    "gap_hse", "gap_dir_hse", "vbm_hse", "cbm_hse", "efermi_hse",
    # effective masses
    "emass_vbm", "emass_cbm",
    # electronic polarizability (screening proxy - physically tied to GW)
    "alphax_el", "alphay_el", "alphaz_el",
    # lattice polarizability (262/339, NaN elsewhere)
    "alphax_lat", "alphay_lat", "alphaz_lat",
    # plasma frequency (335/339)
    "plasmafrequency_x", "plasmafrequency_y",
    # geometry / electrostatics
    "thickness", "evacdiff", "dipz",
    # thermodynamics / stability
    "ehull", "hform", "minhessianeig",
    # symmetry
    "number", "lgnum",
]

TARGET_KEYS = ["gap_gw", "gap_dir_gw", "vbm_gw", "cbm_gw"]


def structure_features(folder):
    s = json.loads((folder / "structure.json").read_text())["1"]
    Z = np.asarray(s["numbers"], dtype=float)
    cell = np.asarray(s["cell"], dtype=float)
    area = float(np.linalg.norm(np.cross(cell[0], cell[1])))
    return {
        "n_atoms": len(Z),
        "n_elements": len(set(Z.tolist())),
        "mean_Z": Z.mean(),
        "std_Z": Z.std(),
        "min_Z": Z.min(),
        "max_Z": Z.max(),
        "area": area,
        "area_per_atom": area / len(Z),
        "energy_per_atom": float(s["energy"]) / len(Z),
    }


def main():
    rows = []
    for gw_path in sorted(MATERIALS_ROOT.rglob("results-asr.gw.json")):
        folder = gw_path.parent
        data = json.loads((folder / "data.json").read_text())
        row = {"uid": data.get("uid"),
               "folder": str(folder.relative_to(MATERIALS_ROOT)),
               "has_hse_json": (folder / "hse.json").exists()}
        for k in DATA_KEYS + TARGET_KEYS:
            row[k] = data.get(k, np.nan)
        row.update(structure_features(folder))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} materials, {df.shape[1]} columns -> {OUT}")
    print("NaN counts (nonzero only):")
    nn = df.isna().sum()
    print(nn[nn > 0].to_string())
    print(f"gap_gw: min {df.gap_gw.min():.3f}, median {df.gap_gw.median():.3f}, "
          f"max {df.gap_gw.max():.3f} eV")


if __name__ == "__main__":
    main()
