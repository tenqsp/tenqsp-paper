# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Figure 7, the simulations: the spheroid under three schedules with the same average plasma
level (infusion, a bolus every 7, every 21), with a linear kill and two saturable ones, at
two more penetration speeds, and without the depth axis. Writes results/fig7_<mode>/.

    python figures/fig7_runs.py            # the paper's grid: about 3 min on 8 cores
    python figures/fig7_runs.py --quick    # 20 x 50 cells: under 1 min
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import spheroid as sp

ISAT = {"0": 0.0, "0.25": 0.25, "1": 1.0}  # 1/EC50: linear kill, EC50 = 4, EC50 = 1


def runs():
    """(model, isat, schedule, epsd) of every run in the figure."""
    out = [("spheroid_pk", i, s, 1.0) for i in ISAT for s in sp.SCHEDULES]
    out += [("spheroid_pk", "0.25", s, e) for s in ("7", "21") for e in (0.1, 5.0)]
    out += [("spheroid_nodepth", i, s, None) for i in ("0", "0.25") for s in sp.SCHEDULES]
    return out


def one(job):
    name, isat, schedule, epsd, mode = job
    m = sp.model(name, mode)
    params = {"isat": ISAT[isat]} | ({} if epsd is None else {"epsd": epsd})
    times = np.linspace(0, sp.T, 1001)
    t0 = time.perf_counter()
    sol = m.solve((0, sp.T), times=times, regimen=sp.regimen(schedule), params=params, **sp.SOLVER)
    keep = {"t": times, "total": np.asarray(sol["total"]), "mean_x": np.asarray(sol["mean_x"])}
    if name == "spheroid_pk":
        keep["n"] = np.asarray(sol["n"])[::100]  # every 100 time units
    tag = f"{name}_{isat}_{schedule}" + ("" if epsd is None else f"_{epsd:g}")
    np.savez_compressed(sp.results_dir(mode) / f"{tag}.npz", **keep)
    return f"{tag}: {time.perf_counter() - t0:.0f} s, burden at t = {sp.T:g}: {keep['total'][-1]:.3g}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="20 x 50 cells instead of 40 x 100")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    mode = "quick" if a.quick else "full"
    with ProcessPoolExecutor(a.jobs) as ex:
        for line in ex.map(one, [r + (mode,) for r in runs()]):
            print(line, flush=True)
