"""Record which TabPFN actually produced a number.

Reviewer feedback on the draft was that the TabPFN version is never stated. It
is worse than an omission: the cloud backend selects the model server-side
(`model_path="auto"`), so the client version alone does not pin the estimator,
and two runs months apart can differ without anything in the repository
changing. The local backend is pinned by the installed package and is the
reproducible path.

Every result-writing script calls `stamp()` and the resulting JSON sits next to
the CSV it describes. Report `backend`, the version fields, and - for cloud -
the run date, because that is what identifies the served model.

    from provenance import stamp
    stamp("delta_cv_results_cloud_nomag.csv", backend="cloud",
          config=dict(train_rows=10000, folds=5, seed=0))
"""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone

import paths

PACKAGES = ["tabpfn", "tabpfn-client", "scikit-learn", "numpy", "pandas",
            "scipy", "torch", "ase", "matplotlib"]


def _version(pkg):
    try:
        import importlib.metadata as md
        return md.version(pkg)
    except Exception:
        return None


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=paths.ROOT, capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def cloud_estimator_settings(model_version=None):
    """The client-side settings that identify the model actually used."""
    import tabpfn_backend
    model_version = model_version or tabpfn_backend.DEFAULT_VERSION
    try:
        r = tabpfn_backend.make_regressor("cloud", model_version)()
        out = {k: getattr(r, k) for k in
               ("model_path", "n_estimators", "paper_version", "random_state",
                "softmax_temperature")
               if hasattr(r, k)}
    except Exception as exc:
        out = {"unavailable": str(exc)}
    out["model_version"] = model_version
    out["pinned"] = model_version != "auto"
    out["summary"] = tabpfn_backend.describe("cloud", model_version)
    return out


def describe(backend="cloud", config=None, model_version=None):
    rec = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": backend,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "packages": {p: _version(p) for p in PACKAGES},
        "config": config or {},
    }
    if backend == "cloud":
        rec["cloud_estimator"] = cloud_estimator_settings(model_version)
        pinned = rec["cloud_estimator"].get("pinned")
        rec["note"] = (
            "Cloud model pinned by name; the client resolves it to a checkpoint."
            if pinned else
            "model_path='auto': the server chose the model. This run is "
            "identified by client version AND run date, not by version alone. "
            "Pass --model-version to pin it."
        )
    else:
        rec["note"] = ("Local backend: the estimator is pinned by the installed "
                       "tabpfn package version recorded above.")
    return rec


def stamp(result_filename, backend="cloud", config=None, model_version=None):
    """Write <result>.provenance.json beside the result file. Returns the record."""
    rec = describe(backend=backend, config=config, model_version=model_version)
    out = paths.DATA / (str(result_filename).rsplit(".", 1)[0] + ".provenance.json")
    out.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    return rec


if __name__ == "__main__":
    backend = sys.argv[1] if len(sys.argv) > 1 else "cloud"
    mv = sys.argv[2] if len(sys.argv) > 2 else None
    print(json.dumps(describe(backend=backend, model_version=mv),
                     indent=2, ensure_ascii=False))
