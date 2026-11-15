# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Figure 5 A-C, the fits: every simulated study fitted with every chain length; writes
results/fig5/fits.csv (one row per study and chain) and results/fig5/study1.csv.

    python figures/fig5_runs.py               # 20 studies x 9 chain lengths
    python figures/fig5_runs.py --studies 2   # a quick check
"""

import argparse
import multiprocessing
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import transit_fit as tr


def one(job):
    seed, n = job
    data = tr.study(seed)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # warnings of weakly determined fits, for some chains
        r = tr.tf.fit(tr.problem(n, data))
    row = {"study": seed, "N": n, "m2ll": float(r.m2ll), "status": r.status,
           "seconds": time.perf_counter() - t0} | r.estimates
    table = r.table().set_index("quantity")
    for k in tr.TRUE:
        row[f"{k}_lo"], row[f"{k}_hi"] = (float(table.loc[k, "CI 2.5 %"]), float(table.loc[k, "CI 97.5 %"]))
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    tr.RESULTS.mkdir(parents=True, exist_ok=True)
    tr.study(1).to_csv(tr.RESULTS / "study1.csv", index=False)
    jobs = [(s, n) for s in range(1, a.studies + 1) for n in tr.CHAINS]
    rows = []
    t0 = time.perf_counter()
    # fresh worker processes (forking after JAX has started can hang), each on one thread
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(a.jobs, mp_context=ctx) as ex:
        for row in ex.map(one, jobs):
            rows.append(row)
            print(f"study {row['study']:2d}, N = {row['N']:2d}: -2LL {row['m2ll']:9.2f}, MTT {row['MTT']:.3f}, "
                  f"ka {row['ka']:.3f}, CL {row['CL']:.3f}, V {row['V']:.2f} ({row['seconds']:.0f} s)", flush=True)
    pd.DataFrame(rows).to_csv(tr.RESULTS / "fits.csv", index=False)
    print(f"{len(rows)} fits in {(time.perf_counter() - t0) / 60:.0f} min on {a.jobs} processes ({os.cpu_count()} cores)")
