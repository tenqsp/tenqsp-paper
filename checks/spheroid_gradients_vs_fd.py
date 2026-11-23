# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Check the dose derivatives of Figure 6F against central finite differences.

Figure 6F's derivatives come from one reverse-mode (record and replay) computation in
figures/fig7_gradients.py. The same function computes them here, on the quick grid (20 x 50 cells)
by default, and they are compared with central differences of the scipy backend at a tight
tolerance, for the time (moved 0.5 earlier and later) and the amount (1 % smaller and larger) of
five boluses (one every 21, EC50 = 4), for all four objectives. The areas are integrated by the
solver, as in fig7_gradients.py. Writes results/fig7_<mode>/fd_check.csv.

    python checks/spheroid_gradients_vs_fd.py           # quick grid: about 15 min, 20 GB
    python checks/spheroid_gradients_vs_fd.py --full    # the paper's grid
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "figures"))
import spheroid as sp
from fig7_gradients import AREAS, OBJECTIVES, gradients

DOSES = (1, 24, 44, 46, 48)  # the first, one in the middle and the last few
H_TIME, H_AMOUNT = 0.5, 0.01  # the time step, and the relative step of the amount
H_FIRST = 0.004  # dose 1 is at t = 0: a one-sided, first-order difference needs a small step


def objectives(job):
    """The four objectives of Figure 6F, by the scipy backend at a tight tolerance."""
    mode, what, k, shift = job
    m = sp.model("spheroid_pk", mode, extra=AREAS)
    dt = sp.dose_times("21")
    amounts = [sp.CAVG * sp.KE * 21] * len(dt)
    if what == "time":
        dt[k - 1] += shift
    elif what == "amount":
        amounts[k - 1] *= 1 + shift
    reg = sp.tq.bolus("P1", amounts, times=dt)
    sol = m.solve((0, sp.T), times=[0, sp.T], regimen=reg, params={"isat": 0.25}, rtol=1e-9, atol=1e-25)
    return np.array([np.asarray(sol["auc"])[-1] / sp.T, np.asarray(sol["auclog"])[-1] / sp.T,
                     np.log(np.asarray(sol["total"])[-1]), np.asarray(sol["mean_x"])[-1]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    mode = "full" if ap.parse_args().full else "quick"
    # dose 1 is at the start: its time can only move later (a one-sided difference)
    jobs = [(mode, "none", 0, 0.0)]
    jobs += [(mode, "time", k, s) for k in DOSES if k > 1 for s in (H_TIME, -H_TIME)]
    jobs += [(mode, "time", 1, H_FIRST)] if 1 in DOSES else []
    jobs += [(mode, "amount", k, s) for k in DOSES for s in (H_AMOUNT, -H_AMOUNT)]
    with ProcessPoolExecutor(len(jobs)) as ex:
        fut = ex.map(objectives, jobs)
        J, y, x0, _ = gradients(mode)  # reverse mode, in this process, while the scipy runs go on
        v = dict(zip(((w, k, s) for _, w, k, s in jobs), fut, strict=True))
    n = len(sp.dose_times("21"))
    amount = sp.CAVG * sp.KE * 21
    rows, ok = ["objective,input,reverse,finite_differences,relative_difference,ok"], True
    for k in DOSES:
        if k == 1:
            fd_t = (v[("time", 1, H_FIRST)] - v[("none", 0, 0.0)]) / H_FIRST
        else:
            fd_t = (v[("time", k, H_TIME)] - v[("time", k, -H_TIME)]) / (2 * H_TIME)
        fd_a = (v[("amount", k, H_AMOUNT)] - v[("amount", k, -H_AMOUNT)]) / (2 * H_AMOUNT * amount)
        for what, col, fd in (("time", k - 1, fd_t), ("amount", n + k - 1, fd_a)):
            for i, name in enumerate(OBJECTIVES):
                a, b = float(J[i, col]), float(fd[i])
                floor = 1e-4 * max(abs(float(y[i])), 1e-2) / (1 if what == "time" else amount)
                rel = abs(a - b) / max(abs(b), floor)
                good = rel <= 2e-3
                ok &= good
                rows.append(f"{name},dose {k} {what},{a:.6e},{b:.6e},{rel:.2e},{int(good)}")
    out = sp.results_dir(mode) / "fd_check.csv"
    out.write_text("\n".join(rows) + f"\n# tenqsp {sp.tq.__version__}\n")
    print("\n".join(rows))
    sys.exit(0 if ok else 1)
