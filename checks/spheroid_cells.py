# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Check the Figure 6 results when the model counts whole cells.

The model has no mutation along the resistance axis x, so every resistance level at the end is
already present at the start, in the Gaussian tails of n(0). For a real spheroid of N0 cells those
tails hold far less than one cell beyond some resistance level, and during a deep response the
whole tumor can fall below one cell. Here the nine runs of Figure 6 (three kills, three schedules)
are repeated for N0 = 1e5 and 1e6 cells with both effects counted:

- the initial tails are cut where a resistance cell (width 1/100) of the whole spheroid would
  hold less than one cell: n(0) = max(0.1 exp(-(x - 0.5)^2/0.005) - n1, 0), with n1 the density
  of one cell;
- the tumor is an extinction group that is set to zero if its total falls below one cell.

For each run: the deepest response, the burden at the end, the mean resistance at the end and
whether the tumor went extinct, next to the paper's continuous run. Writes
results/spheroid_cells/cells.csv.

    python checks/spheroid_cells.py      # 18 runs on the paper's grid, about 10 min on 18 cores
"""

import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "figures"))
import spheroid as sp

ISAT = {"linear kill": 0.0, "EC50 4": 0.25, "EC50 1": 1.0}
SIZES = (1e5, 1e6)
TOTAL0 = 0.1 * np.sqrt(np.pi * 0.005) * 4 * np.pi / 3  # the amount at the start (unit sphere)
FIRST_CORE = int(os.environ.get("FIRST_CORE", "0"))


def _pin(cores):
    """Pin this worker to one core of its own: JAX's thread pools spin when cores are shared."""
    os.sched_setaffinity(0, {cores.get()})


def text(size):
    """The paper's model with whole cells counted, for a spheroid of `size` cells."""
    nx = sp.GRIDS["full"][1]
    one_cell = TOTAL0 / size  # the amount of one cell
    n1 = one_cell / (4 * np.pi / 3) * nx  # the density of one cell in one resistance cell, uniform in r
    t = (sp.MODELS / "spheroid_pk.tq").read_text()
    t = t.replace("n(0) = 0.1*exp(-(x - 0.5)^2/0.005)", f"n(0) = max(0.1*exp(-(x - 0.5)^2/0.005) - {n1:.6g}, 0)")
    assert f"- {n1:.6g}, 0)" in t
    return t + f"\nextinction tumor: n below {one_cell:.6g}\n"


def one(job):
    size, kill, schedule = job
    import tenqsp as tq

    nr, nx = sp.GRIDS["full"]
    t = text(size).replace("n=40 geometry", f"n={nr} geometry").replace("n=100 ", f"n={nx} ")
    m = tq.Model.from_text(t)
    sol = m.solve((0, sp.T), times=np.linspace(0, sp.T, 1001), regimen=sp.regimen(schedule),
                  params={"isat": ISAT[kill]}, **sp.SOLVER)  # fmt: skip
    total = np.asarray(sol["total"])
    extinct = bool(len(getattr(sol, "extinctions", []) or []))
    mean_x = np.asarray(sol["mean_x"])
    return (float(total.min()), float(total[-1]), float(mean_x[-1]) if total[-1] > 0 else float("nan"), extinct,
            float(np.nanmax(np.where(total > 0, mean_x, np.nan))))


def continuous(kill, schedule):
    key = {"linear kill": "0", "EC50 4": "0.25", "EC50 1": "1"}[kill]
    a = np.load(sp.RESULTS / "fig7_full" / f"spheroid_pk_{key}_{schedule}_1.npz")
    return float(a["total"].min()), float(a["total"][-1]), float(a["mean_x"][-1])


if __name__ == "__main__":
    jobs = [(n, k, s) for n in SIZES for k in ISAT for s in sp.SCHEDULES]
    ctx = multiprocessing.get_context("spawn")
    cores = ctx.Queue()
    for c in range(FIRST_CORE, FIRST_CORE + len(jobs)):
        cores.put(c)
    with ProcessPoolExecutor(len(jobs), mp_context=ctx, initializer=_pin, initargs=(cores,)) as ex:
        v = dict(zip(jobs, ex.map(one, jobs), strict=True))
    out = ROOT / "results" / "spheroid_cells"
    out.mkdir(parents=True, exist_ok=True)
    import tenqsp as tq

    rows = [("cells,kill,schedule,deepest,end,mean_resistance_end,extinct,"
             "continuous_deepest,continuous_end,continuous_mean_resistance_end,tenqsp")]
    for (n, k, s), (d, e, x, ext, _) in v.items():
        cd, ce, cx = continuous(k, s)
        rows.append(f"{n:.0e},{k},{s},{d:.6g},{e:.6g},{x:.6f},{int(ext)},{cd:.6g},{ce:.6g},{cx:.6f},{tq.__version__}")
    (out / "cells.csv").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
