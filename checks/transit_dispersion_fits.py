# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Positive control for Figure 3 A-C: the 20 simulated studies fitted with the true structure.

The studies of figures/fig3_runs.py are simulated from transport with dispersion (`using sharp` on
200 cells). Here each is fitted with that same structure on fewer cells, estimating MTT, ka, CL, V
and the dispersion, with the full checks of tenqsp-fit. The coverage of these intervals is the
reference against which the chain fits are read. Writes results/fig3/dispersion_fits.csv.

    python checks/transit_dispersion_fits.py               # 20 studies on 40 cells
    python checks/transit_dispersion_fits.py --studies 1 --cells 20 40 80   # grid check
"""

import argparse
import multiprocessing
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "figures"))
import transit_fit as tr

PARAMS = ("MTT", "ka", "CL", "V", "disp")


def _pin(cores):
    """Pin this worker to one core of its own: JAX's thread pools spin when cores are shared."""
    os.sched_setaffinity(0, {cores.get()})


def one(job):
    seed, n = job
    data = tr.study(seed)
    p = tr.tf.Problem(tr.transit_model(n, dispersion=True), name=f"dispersion{n}")
    p.data(data)
    p.unit("ID")
    p.observe("conc", "DV")
    p.estimate("MTT", 1.0)
    p.estimate("ka", 2.0, bounds=(0.3, 100))
    p.estimate("CL", 2.0)
    p.estimate("V", 20.0)
    p.estimate("disp", 0.05, bounds=(1e-4, 1))
    p.fix("D", 100)
    p.error("conc", tr.tf.combined(0.05, 0.2))
    p.censor("conc", below=tr.LLOQ, where="BLQ == 1")
    p.method("optimize")
    p.options(starts=4, seed=1, progress="none")  # the full checks (tolerance, gradient)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = tr.tf.fit(p)
    row = {"study": seed, "cells": n, "m2ll": float(r.m2ll), "status": r.status,
           "seconds": time.perf_counter() - t0} | r.estimates
    table = r.table().set_index("quantity")
    for k in PARAMS:
        row[f"{k}_lo"], row[f"{k}_hi"] = (float(table.loc[k, "CI 2.5 %"]), float(table.loc[k, "CI 97.5 %"]))
    d = r.doctor()
    row["doctor"] = d.headline()
    row["findings"] = " | ".join(f"{f.severity} {f.check}: {f.message}" for f in d.findings)
    row["versions"] = f"tenqsp {tr.tq.__version__}, tenqsp-fit {tr.tf.__version__}"
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", type=int, default=20)
    ap.add_argument("--cells", type=int, nargs="+", default=[40])
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--out", default="dispersion_fits.csv")
    ap.add_argument("--first-core", type=int, default=0)
    a = ap.parse_args()
    jobs = [(s, n) for s in range(1, a.studies + 1) for n in a.cells]
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    rows = []
    ctx = multiprocessing.get_context("spawn")
    cores = ctx.Queue()
    for c in range(a.first_core, a.first_core + a.jobs):
        cores.put(c)
    with ProcessPoolExecutor(a.jobs, mp_context=ctx, initializer=_pin, initargs=(cores,)) as ex:
        for row in ex.map(one, jobs):
            rows.append(row)
            print(f"study {row['study']:2d}, {row['cells']} cells: -2LL {row['m2ll']:.2f}, MTT {row['MTT']:.3f}, "
                  f"ka {row['ka']:.3f}, disp {row['disp']:.4f} ({row['seconds']:.0f} s)", flush=True)
    pd.DataFrame(rows).to_csv(tr.RESULTS / a.out, index=False)
