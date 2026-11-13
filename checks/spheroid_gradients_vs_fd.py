# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Check the exact dose-time derivatives of Figure 7F against central finite differences.

The derivatives of the log burden at the end and of the mean burden over the course with
respect to the time of some boluses (one every 21, EC50 = 4), from TenQSP's forward
sensitivities, against central differences of the scipy backend at a tight tolerance (each
dose moved 0.5 earlier and later). The mean burden comes from the area integrated by the
solver, as in fig7_gradients.py. On the quick
grid (20 x 50 cells) by default: about 8 min on 8 cores; --full: the paper's grid.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "figures"))
import spheroid as sp
from fig7_gradients import AREAS

DOSES = (1, 24, 44, 46, 48)  # the first, one in the middle and the last few
H = 0.5


def objectives(job):
    """The log burden at the end and the mean burden over the course."""
    mode, k, shift = job
    m = sp.model("spheroid_pk", mode, extra=AREAS)
    dt = sp.dose_times("21")
    if k:
        dt[k - 1] += shift
    reg = sp.tq.bolus("P1", sp.CAVG * sp.KE * 21, times=dt)
    sol = m.solve((0, sp.T), times=[0, sp.T], regimen=reg, params={"isat": 0.25},
                  rtol=1e-9, atol=1e-25)
    return np.array([np.log(np.asarray(sol["total"])[-1]), np.asarray(sol["auc"])[-1] / sp.T])


def exact(mode):
    m = sp.model("spheroid_pk", mode, extra=AREAS)
    sim = m.simulator(outputs=["total", "auc"], **sp.SOLVER)
    sc = sp.tq.Scenario(params={"isat": 0.25}, regimen=sp.regimen("21"), times=[0, sp.T])
    f = sim.function(sc, wrt=[f"dose {k} time" for k in DOSES], derivatives="forward")

    def objectives(x):
        o = f(x)
        return jnp.stack([jnp.log(o["total"][-1]), o["auc"][-1] / sp.T])

    return np.asarray(jax.jacfwd(objectives)(f.x0)), np.asarray(objectives(f.x0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    mode = "full" if ap.parse_args().full else "quick"
    # dose 1 is at the start: it can only move later (a one-sided difference)
    jobs = [(mode, 0, 0.0)] + [(mode, k, s) for k in DOSES for s in (H, -H) if k > 1 or s > 0]
    with ProcessPoolExecutor(8) as ex:
        v = dict(zip(((k, s) for _, k, s in jobs), ex.map(objectives, jobs), strict=True))
    fd = [(v[(1, H)] - v[(0, 0.0)]) / H] + [(v[(k, H)] - v[(k, -H)]) / (2 * H) for k in DOSES[1:]]
    J, y = exact(mode)
    ok = True
    for i, name in enumerate(("log burden at the end", "mean burden")):
        print(name)
        floor = 1e-4 * max(abs(y[i]), 1e-2)  # below this a derivative counts as zero
        for k, a, b in zip(DOSES, J[i], (d[i] for d in fd), strict=True):
            good = abs(a - b) <= 2e-3 * max(abs(b), floor)
            ok &= good
            print(f"{'OK ' if good else 'XX '} dose {k:2d}: exact {a: .5e}  finite differences {b: .5e}")
    sys.exit(0 if ok else 1)
