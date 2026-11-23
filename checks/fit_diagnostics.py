# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Goodness of fit for the two population fits of Figure 3 (Supplement S7).

Refits the Friberg model (figures/fig5_friberg.py) and the theophylline data with the chain of
3 (figures/fig5_theoph.py, the lowest -2LL among the chains; here from 12 starts), both by
laplace (FOCE-I), and writes for each (for the chains of 2 and 5, the parameter tables only:
the chain of 3 has no standard errors, its Hessian being indefinite along ka, its spread and MTT):

- results/fit_diagnostics/<name>_parameters.csv: estimates, standard errors, RSE %, 95 %
  intervals, and the shrinkage of each between-unit parameter (NONMEM's SD-based eta shrinkage);
- results/fit_diagnostics/<name>_sdtab.csv and _patab.csv (tenqsp-fit's tables);
- out/figS_gof_<name>.png and .pdf: DV against PRED and IPRED, IWRES against time and IPRED, and
  a visual predictive check (200 simulated studies; 5th, 50th and 95th percentiles).

    python checks/fit_diagnostics.py      # a few minutes
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "figures"))
import fig5_friberg
import fig5_theoph
import tenqsp as tq
import tenqsp_fit as tf
from tenqsp_fit import results as tr

OUT = ROOT / "results" / "fit_diagnostics"


def fits():
    """(name, problem, log scale for the VPC, draw the figure)."""
    yield "friberg", fig5_friberg.problem()[2], True, True
    # the paper's best chain, from 12 starts (the paper's fit used 4; all reach the same point)
    yield "theophylline_chain3", fig5_theoph.problem(3, starts=12, seed=2), False, True
    # its neighbors, within 0.5 of its -2LL: standard errors where the chain of 3 has none
    yield "theophylline_chain2", fig5_theoph.problem(2), False, False
    yield "theophylline_chain5", fig5_theoph.problem(5), False, False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "out").mkdir(exist_ok=True)
    summary = {}
    for name, p, log, draw in fits():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = tf.fit(p)
        t = r.table()
        shrink = getattr(r, "shrinkage", {}) or {}
        t["shrinkage %"] = [100 * shrink[q.split(".")[0]] if q.endswith(".sdlog") and q.split(".")[0] in shrink
                            else float("nan") for q in t.quantity]  # fmt: skip
        t.to_csv(OUT / f"{name}_parameters.csv", index=False)
        tr.write_tables(r, OUT)
        for k in ("sdtab", "patab"):
            (OUT / f"{k}.csv").rename(OUT / f"{name}_{k}.csv")
        summary[name] = {"m2ll": float(r.m2ll), "status": r.status, "doctor": r.doctor().headline(),
                         "findings": [f"{f.severity} {f.check}: {f.message}" for f in r.doctor().findings],
                         "tenqsp": tq.__version__, "tenqsp_fit": tf.__version__}
        print(name, summary[name]["m2ll"], summary[name]["doctor"])
        print(t.round(4).to_string(index=False))
        if not draw:
            continue
        fig = plt.figure(figsize=(13.2, 3.1 + 3.8 + 0.6), layout="constrained")
        top, bottom = fig.subfigures(2, 1, height_ratios=[3.1, 3.8])
        tr.plot_gof(r, fig=top)
        v = tr.vpc(r, n=200, seed=1, log=log, rtol=1e-6, fig=bottom)
        for ext in ("png", "pdf"):
            fig.savefig(ROOT / "out" / f"figS_gof_{name}.{ext}", dpi=200)
        plt.close(fig)
        v.table.to_csv(OUT / f"{name}_vpc.csv", index=False)
        summary[name]["vpc_failed"] = int(v.failed)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
