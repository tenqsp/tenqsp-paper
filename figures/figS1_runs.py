# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Figure S1, the computations: runs TenQSP's 12 tutorials and stores, for every model a
tutorial solves, its description (the IR, as JSON) and, from two of its solutions, the amount of
each field over each of its axes and time, into
results/figS1/<NN>.json (about 5 min, one tutorial per process).

    python figures/figS1_runs.py                    # tutorials of the installed version
    python figures/figS1_runs.py --tutorials DIR    # tutorials in a local folder
    python figures/figS1_runs.py --only 01 05       # some of them

The tutorials come from the source distribution of the installed TenQSP version, as for
Table S2. Every scipy solve a tutorial makes is recorded; for each model (by name) the first solution
and the first on its finest grid are kept, on at most 60 output times.
"""

import argparse
import contextlib
import io
import json
import os
import runpy
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "figS1"
sys.path.insert(0, str(ROOT / "tables"))
KEEP_TIMES = 60


def short(x):
    """Values to 4 significant digits, as lists (the figure needs no more)."""
    return np.vectorize(lambda v: float(f"{v:.4g}"))(np.asarray(x, float)).tolist()


def profiles(sol, keep):
    """Per field, its amount over each continuous axis (summed over the others) and time, with
    the axis's cell faces; a field with no continuous axis over its categories."""
    out = {}
    for name, axes in sol.axes.items():
        u = np.asarray(sol[name])[keep]
        if not axes:
            continue
        cont = [a for a in axes if "faces" in sol.coords[a]]
        widths = [np.diff(sol.coords[a]["faces"]) if a in cont else np.ones(u.shape[i + 1])
                  for i, a in enumerate(axes)]
        out[name] = {}
        for k, a in enumerate(cont):
            v = u
            for i in reversed(range(len(axes))):
                if axes[i] != a:
                    v = np.tensordot(v, widths[i], axes=([i + 1], [0]))
            out[name][a] = {"amount": short(v), "faces": short(sol.coords[a]["faces"])}
        if not cont:
            out[name]["categories"] = {"amount": short(u.reshape(len(keep), -1))}
    return out


def record_one(path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tenqsp
    from tenqsp.backends import scipy_backend

    models = {}
    original = scipy_backend.solve

    def solve(cm, t_span, **kwargs):
        sol = original(cm, t_span, **kwargs)
        name = cm.ir.name
        entry = models.setdefault(name, {"solves": 0})
        entry["solves"] += 1
        size = int(sol.y.shape[1])
        if "first" not in entry or size > entry["finest"]["size"]:
            keep = np.unique(np.linspace(0, len(sol.t) - 1, min(KEEP_TIMES, len(sol.t))).round().astype(int))
            record = {"ir": cm.ir.to_dict(), "size": size, "t": short(np.asarray(sol.t)[keep]),
                      "profiles": profiles(sol, keep)}
            entry.setdefault("first", record)
            entry["finest"] = record
        return sol

    scipy_backend.solve = solve
    path = Path(path)
    os.chdir(path.parent)
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(str(path), run_name="__main__")
    plt.close("all")
    out = {"tutorial": path.name, "tenqsp": tenqsp.__version__,
           "seconds": round(time.perf_counter() - start, 1), "models": models}
    key = path.name[:2]
    (RESULTS / f"{key}.json").write_text(json.dumps(out) + "\n")
    return key, out["seconds"], list(models)


def main():
    from tableS2_runs import sdist_tutorials

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tutorials", type=Path, help="folder with the tutorials (default: from PyPI)")
    parser.add_argument("--only", nargs="*", help="tutorial numbers, e.g. 01 05")
    args = parser.parse_args()
    folder = (args.tutorials or sdist_tutorials()).resolve()
    paths = sorted(folder.glob("[01][0-9]_*.py"))
    if args.only:
        paths = [p for p in paths if p.name[:2] in args.only]
    RESULTS.mkdir(parents=True, exist_ok=True)
    paths.sort(key=lambda p: p.name[:2] not in ("02", "11", "12"))
    with ProcessPoolExecutor(max_workers=len(paths), mp_context=get_context("spawn")) as pool:
        jobs = [pool.submit(record_one, str(p)) for p in paths]
        for job in as_completed(jobs):
            key, seconds, names = job.result()
            print(f"tutorial {key}: {seconds:.0f} s, models {', '.join(names)}", flush=True)


if __name__ == "__main__":
    main()
