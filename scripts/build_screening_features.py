"""Build features for the GW-gap screening release.

Scans ALL C2DB materials and keeps those with a PBE gap entry (~8.7k,
superset of the ~3.4k that also have HSE). Extracts the same feature set as
build_gap_features.py plus a magnetic flag and GW availability. Downstream
tiers: tier 1 = has gap_hse (HSE-informed model), tier 2 = PBE-only.

Output: data/screening_features_all.csv
"""

import json
from multiprocessing import Pool
from pathlib import Path

import paths

import numpy as np
import pandas as pd

from build_gap_features import DATA_KEYS, structure_features, MATERIALS_ROOT

OUT = paths.DATA / "screening_features_all.csv"


def process(folder_str):
    folder = Path(folder_str)
    try:
        data = json.loads((folder / "data.json").read_text())
        if data.get("gap") is None:
            return None
        s = json.loads((folder / "structure.json").read_text())["1"]
        site = np.abs(np.asarray(s.get("magmoms", [0.0]), dtype=float))
        magnetic = bool(abs(s.get("magmom") or 0.0) > 0.01 or site.max() > 0.01
                        or (folder / "magmoms.json").exists())
        row = {"uid": data.get("uid"),
               "folder": str(folder.relative_to(MATERIALS_ROOT)),
               "has_gw": (folder / "results-asr.gw.json").exists(),
               "magnetic": magnetic,
               "gap_gw": data.get("gap_gw", np.nan)}
        for k in DATA_KEYS:
            row[k] = data.get(k, np.nan)
        row.update(structure_features(folder))
        return row
    except Exception as exc:
        return {"uid": "", "folder": str(folder), "error": repr(exc)}


def main():
    folders = sorted(str(p.parent) for p in MATERIALS_ROOT.rglob("data.json"))
    print(f"scanning {len(folders)} materials")
    with Pool(8) as pool:
        rows = [r for r in pool.imap_unordered(process, folders, chunksize=64)
                if r is not None]
    errors = [r for r in rows if r.get("error")]
    rows = [r for r in rows if not r.get("error")]
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} materials with PBE gap entry -> {OUT} ({len(errors)} errors)")
    print(f"  has_gw: {df.has_gw.sum()}, magnetic: {df.magnetic.sum()}, "
          f"has gap_hse: {df.gap_hse.notna().sum()}")
    base = df[~df.has_gw & ~df.magnetic & (df.gap > 0.05)]
    print(f"  non-magnetic gapped, no GW: {len(base)}")
    print(f"    tier 1 (with HSE): {(base.gap_hse > 0.05).sum()}")
    print(f"    tier 2 (PBE-only): {base.gap_hse.isna().sum()}")


if __name__ == "__main__":
    main()
