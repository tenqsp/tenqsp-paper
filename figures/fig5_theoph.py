# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Figure 5 D, the fits: theophylline in 12 subjects (R's Theoph data, data/theoph.csv),
population fits (method laplace, FOCE-I) with first-order absorption and with a transit
chain of N compartments before it, N = 1 ... 20 with the mean transit time estimated. ka
and CL vary between subjects as in tenqsp-fit's tutorial 1. Writes results/fig5/theoph.csv.

    python figures/fig5_theoph.py
"""

import multiprocessing
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import tenqsp as tq
import tenqsp_fit as tf
from transit_fit import RESULTS, ROOT

CHAINS = (0, 1, 2, 3, 5, 10, 20)  # 0: first-order absorption, no transit


def model(n):
    if n == 0:
        return tq.Model.from_text("""
model theoph_fo
param ka=1.6, CL=0.04, ke=0.086, D=4
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - ke*centr
depot(0) = D
observable conc = centr*ke/CL
""")
    return tq.Model.from_text(f"""
model theoph_chain{n}
axis z in [0, 1] n={n} kind=one_way
param MTT=0.3, ka=1.6, CL=0.04, ke=0.086, D=4
field u(z)
d_t(u) = -d_z(u/MTT)
    at z=1: outflow
d_t(depot) = u(z=1)/MTT - ka*depot
d_t(centr) = ka*depot - ke*centr
u(0) = D*point(z, at={0.5 / n})
observable conc = centr*ke/CL
""")


def one(n):
    p = tf.Problem(model(n), name=f"theoph_{n}")
    p.data(pd.read_csv(ROOT / "data" / "theoph.csv"), ID="Subject", TIME="Time", DV="conc")
    p.unit("ID")
    p.observe("conc", "DV")
    p.define("D", "Dose")  # mg/kg, a column of the data
    p.estimate("ka", tf.lognormal(median=1.5, sdlog=0.5))
    p.estimate("CL", tf.lognormal(median=0.04, sdlog=0.5))
    p.estimate("ke", 0.08)
    if n:
        p.estimate("MTT", 0.3, bounds=(1e-3, 5))
    p.error("conc", tf.additive(1))
    p.method("laplace")
    p.options(starts=4, seed=1, progress="none")
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = tf.fit(p)
    return {"N": n, "m2ll": float(r.m2ll), "status": r.status, "seconds": time.perf_counter() - t0} | r.estimates


if __name__ == "__main__":
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    RESULTS.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(len(CHAINS), mp_context=multiprocessing.get_context("spawn")) as ex:
        rows = list(ex.map(one, CHAINS))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "theoph.csv", index=False)
    print(out.round(4).to_string(index=False))
