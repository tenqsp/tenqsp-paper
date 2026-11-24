# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Figure 6F, which doses matter: exact derivatives of the tumor burden with respect to the
time and the amount of each of the 48 boluses (one every 21, EC50 = 4), from one run with
reverse derivatives (one backward pass per objective). Objectives: the mean burden over the course (the area under the
burden curve divided by its length), the mean log burden, and the burden and the mean
resistance level at the end. The areas are integrated by the solver as two extra states:
a sum over output times would move with each dose time (the burden curve has a kink at
every bolus) and give derivatives that reflect the sum, not the burden. Writes results/fig6_<mode>/gradients.npz.

    python figures/fig6_gradients.py            # the paper's grid: about 30 min, 80 GB
    python figures/fig6_gradients.py --quick    # 20 x 50 cells: about 6 min, 20 GB

checks/spheroid_gradients_vs_fd.py compares these derivatives with finite differences.
"""

import argparse
import resource
import time

import jax
import jax.numpy as jnp
import numpy as np
import spheroid as sp

OBJECTIVES = ("mean burden", "mean log burden", "log burden at the end", "resistance at the end")
AREAS = """
d_t(auc) = int_r(rho)            # the area under the burden curve
d_t(auclog) = log(int_r(rho))    # the area under the log burden
"""


def gradients(mode, schedule="21", isat=0.25):
    m = sp.model("spheroid_pk", mode, extra=AREAS)
    sim = m.simulator(outputs=["total", "mean_x", "auc", "auclog"], **sp.SOLVER)
    sc = sp.tq.Scenario(params={"isat": isat}, regimen=sp.regimen(schedule), times=[sp.T])
    k = len(sp.dose_times(schedule))
    wrt = [f"dose {i + 1} time" for i in range(k)] + [f"dose {i + 1} amount" for i in range(k)]
    f = sim.function(sc, wrt=wrt, derivatives="reverse")
    def objectives(x):
        o = f(x)
        return jnp.stack([o["auc"][-1] / sp.T, o["auclog"][-1] / sp.T,
                          jnp.log(o["total"][-1]), o["mean_x"][-1]])

    def with_value(x):
        e = objectives(x)
        return e, e

    # one run forward, then one pass back per objective (the value rides along as aux)
    J, y = jax.jacrev(with_value, has_aux=True)(f.x0)
    return np.asarray(J), np.asarray(y), np.asarray(f.x0), np.array(sp.dose_times(schedule))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="20 x 50 cells instead of 40 x 100")
    mode = "quick" if ap.parse_args().quick else "full"
    t0 = time.perf_counter()
    J, y, x0, dt = gradients(mode)
    secs = time.perf_counter() - t0
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # kB on Linux
    np.savez(sp.results_dir(mode) / "gradients.npz", J=J, y=y, x0=x0, dose_times=dt,
             objectives=np.array(OBJECTIVES), seconds=secs, peak_gb=peak_gb)
    k = len(dt)
    print(f"{2 * k} derivatives of {len(OBJECTIVES)} objectives: {secs:.0f} s, peak memory {peak_gb:.1f} GB")
    for i, name in enumerate(OBJECTIVES):
        print(f"{name} = {y[i]:.4g}")
        print("   relative change per unit dose time:", np.round(J[i, :k] / abs(y[i]), 5))
        print("   relative change per 1 % of each dose:", np.round(J[i, k:] * x0[k:] / 100 / abs(y[i]), 6))
