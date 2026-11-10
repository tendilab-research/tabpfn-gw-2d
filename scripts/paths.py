"""Where the data lives.

The internal working copy of this project hardcodes absolute paths in every
script. That is deliberate there and wrong for a public release, so the release
resolves them once, here.

Resolution order:
  1. the environment variables PBEHSE_DATA / PBEHSE_FIGURES / PBEHSE_MATERIALS
  2. <repo root>/data, <repo root>/figures, <repo root>/materials

`materials/` holds the raw C2DB dump (structure.json, data.json, hse.json and
results-asr.gw.json per entry). It is NOT redistributed with this release;
obtain it from https://c2db.fysik.dtu.dk and point PBEHSE_MATERIALS at it.
Only the build_*.py stage needs it - every experiment runs from data/ alone.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _resolve(var, default):
    return Path(os.getenv(var) or (ROOT / default)).expanduser()


DATA = _resolve("PBEHSE_DATA", "data")
FIGS = _resolve("PBEHSE_FIGURES", "figures")
MATERIALS_ROOT = _resolve("PBEHSE_MATERIALS", "materials")

DATA.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
