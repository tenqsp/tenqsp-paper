# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Check how the Figure 6 results depend on the grid: depth x resistance cells.

The nine runs of Figure 6 with the depth axis (linear kill, EC50 = 4 and EC50 = 1, under the
three schedules) are repeated on 20 x 50, 40 x 100 (the paper's grid) and 80 x 200 cells, and
on 80 x 100 and 40 x 200 to tell which axis matters. For each run: the deepest response (the
smallest burden over the course), the burden at the end and the mean resistance at the end,
each compared with 80 x 200. Writes results/spheroid_grids/grids.csv.

    python checks/spheroid_grids.py      # about 30 min on 9 cores, 3 GB per run at 80 x 200
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "figures"))
import spheroid as sp

GRIDS = ((20, 50), (40, 100), (80, 100), (40, 200), (80, 200))
FINE = (80, 200)
ISAT = {"linear kill": 0.0, "EC50 4": 0.25, "EC50 1": 1.0}


def one(job):
    (nr, nx), kill, schedule = job
    sp.GRIDS[f"{nr}x{nx}"] = (nr, nx)
    sol = sp.model("spheroid_pk", f"{nr}x{nx}").solve(
        (0, sp.T), times=np.linspace(0, sp.T, 1001), regimen=sp.regimen(schedule),
        params={"isat": ISAT[kill]}, **sp.SOLVER)  # fmt: skip
    total = np.asarray(sol["total"])
    return float(total.min()), float(total[-1]), float(np.asarray(sol["mean_x"])[-1])


if __name__ == "__main__":
    jobs = [(g, k, s) for g in GRIDS for k in ISAT for s in sp.SCHEDULES]
    jobs.sort(key=lambda j: -j[0][0] * j[0][1])  # the largest grids first
    with ProcessPoolExecutor(9) as ex:
        v = dict(zip(jobs, ex.map(one, jobs), strict=True))
    out = ROOT / "results" / "spheroid_grids"
    out.mkdir(parents=True, exist_ok=True)
    rows = ["depth_cells,resistance_cells,kill,schedule,deepest,end,mean_resistance_end"]
    rows += [f"{g[0]},{g[1]},{k},{s},{d:.6g},{e:.6g},{x:.6f}" for (g, k, s), (d, e, x) in v.items()]
    (out / "grids.csv").write_text("\n".join(rows) + "\n")
    print("largest difference from 80 x 200 over the nine runs:")
    for g in GRIDS:
        if g == FINE:
            continue
        rel = np.array([[abs(v[(g, k, s)][i] / v[(FINE, k, s)][i] - 1) for i in (0, 1)]
                        for k in ISAT for s in sp.SCHEDULES])  # fmt: skip
        dx = max(abs(v[(g, k, s)][2] - v[(FINE, k, s)][2]) for k in ISAT for s in sp.SCHEDULES)
        print(f"  {g[0]} x {g[1]}: deepest response {100 * rel[:, 0].max():.2g}%, "
              f"end burden {100 * rel[:, 1].max():.2g}%, mean resistance {dx:.2g}")
