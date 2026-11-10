"""One place that decides which TabPFN answers.

The cloud client's default is `model_path="auto"`, which defers the choice to
the server. That is convenient and unreproducible: the served model can change
without anything here changing, so a result carries no record of what produced
it. Every cloud call in this release goes through `make_regressor()` instead,
which pins a named model version and reports it.

Named versions the installed client accepts:

    v2, v2.5, v2.6, v3      pinned; resolved client-side to a checkpoint name
    auto                    server picks. Kept only to reproduce older runs.

**The committed results predating this module were produced with `auto`**, so
their model version is not recorded and cannot be recovered after the fact.
`identify_committed_model()` re-runs a cheap task under each pinned version and
reports which one matches, which is the only way to find out.

    from tabpfn_backend import make_regressor, MODEL_VERSIONS
    make_tabpfn = make_regressor(backend="cloud", model_version="v3")
    reg = make_tabpfn()
"""

MODEL_VERSIONS = {
    "v2": "v2_default",
    "v2.5": "v2.5_default",
    "v2.6": "v2.6_default",
    "v3": "v3_default",
    "auto": "auto",          # server-side choice; not reproducible
}

DEFAULT_VERSION = "v3"


def resolve(model_version):
    if model_version not in MODEL_VERSIONS:
        raise ValueError(
            f"unknown model_version {model_version!r}; "
            f"choose from {sorted(MODEL_VERSIONS)}")
    return MODEL_VERSIONS[model_version]


def make_regressor(backend="cloud", model_version=DEFAULT_VERSION,
                   random_state=0, **kwargs):
    """Return a zero-argument factory producing a configured TabPFNRegressor.

    The local backend ignores `model_version`: its estimator is pinned by the
    installed `tabpfn` package version, which is the reproducible path.
    """
    if backend == "cloud":
        from tabpfn_client import TabPFNRegressor
        path = resolve(model_version)
        opts = dict(random_state=random_state, **kwargs)
        if path != "auto":
            opts["model_path"] = path
        return lambda: TabPFNRegressor(**opts)

    from tabpfn import TabPFNRegressor
    return lambda: TabPFNRegressor(device="cpu", random_state=random_state,
                                   **kwargs)


def describe(backend, model_version):
    """A one-line provenance string to print and to store beside results."""
    if backend != "cloud":
        try:
            import importlib.metadata as md
            return f"local tabpfn=={md.version('tabpfn')} (pinned by package)"
        except Exception:
            return "local tabpfn (version unknown)"
    try:
        import importlib.metadata as md
        client = md.version("tabpfn-client")
    except Exception:
        client = "?"
    path = resolve(model_version)
    if path == "auto":
        return (f"cloud tabpfn-client=={client}, model_path='auto' "
                f"(SERVER-SIDE CHOICE - not reproducible, identified only by run date)")
    return f"cloud tabpfn-client=={client}, model_path='{path}' (pinned)"


def add_arguments(parser):
    """Attach the standard backend flags to an argparse parser."""
    parser.add_argument("--backend", choices=["cloud", "local"], default="cloud")
    parser.add_argument("--model-version", choices=sorted(MODEL_VERSIONS),
                        default=DEFAULT_VERSION,
                        help="cloud model to pin; 'auto' defers to the server "
                             "and is not reproducible (default: %(default)s)")
    return parser


def identify_committed_model(task="gap_gw", versions=("v2", "v2.5", "v2.6", "v3"),
                             reference=0.0837, drop_magnetic=True):
    """Which pinned version reproduces a committed number?

    Re-runs the cheap scalar-gap cross-validation under each version and prints
    the MAE beside the committed reference. Answers the question that `auto`
    made unanswerable. Costs a few cloud calls per version.
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_absolute_error
    import paths

    df = pd.read_csv(paths.DATA / "gap_gw_features.csv")
    if drop_magnetic:
        spin = pd.read_csv(paths.DATA / "spin_features.csv").set_index("uid")
        keep = ~spin.loc[df.uid, "is_magnetic"].to_numpy(bool)
        df = df[keep]
    drop = ["gap_gw", "gap_dir_gw", "vbm_gw", "cbm_gw", "uid", "folder",
            "has_hse_json"]
    feat = [c for c in df.columns if c not in drop]
    X = df[feat].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    X[~np.isfinite(X)] = np.nan
    y = df["gap_gw"].to_numpy(float)

    print(f"n = {len(y)} materials, {X.shape[1]} features, "
          f"committed reference {reference:.4f} eV\n")
    for v in versions:
        make = make_regressor("cloud", v)
        maes = []
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
            m = make(); m.fit(X[tr], y[tr])
            maes.append(mean_absolute_error(y[te], np.asarray(m.predict(X[te]))))
        mae = float(np.mean(maes))
        print(f"  {v:5s} {describe('cloud', v):70s} MAE {mae:.4f}  "
              f"delta {mae - reference:+.4f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "identify":
        identify_committed_model()
    else:
        for v in sorted(MODEL_VERSIONS):
            print(f"{v:6s} -> {describe('cloud', v)}")
