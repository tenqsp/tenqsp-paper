# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Figure 5 E, the fit: a population fit of the Friberg myelosuppression model with
maturation written as an axis (models/friberg_maturation.tq), to a simulated dose-ranging
study (20 patients at 60, 90 or 120 mg, 8 counts each; circ0 and slope vary between patients,
lognormal sdlog 0.3; 15 % error), by method laplace (FOCE-I), as in tenqsp-fit's tutorial 4.
Writes results/fig5/friberg_estimates.csv, friberg_data.csv and friberg_time.json (the fit's
wall time; run on an idle machine for the paper's number) (a few minutes).

    python figures/fig5_friberg.py
"""

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import tenqsp as tq
import tenqsp_fit as tf
from transit_fit import RESULTS, ROOT

TRUE = {"MTT": 100.0, "gamma": 0.17, "circ0": 5.0, "slope": 0.5}
OMEGA = 0.3
DAYS = np.array([0, 3, 7, 9, 11, 14, 18, 21])


def problem():
    """The simulated study and its fit problem, as in the paper."""
    model = tq.Model.from_text((ROOT / "models" / "friberg_maturation.tq").read_text())
    rng = np.random.default_rng(1)
    rows = []
    for i in range(1, 21):
        dose = [60, 90, 120][i % 3]
        own = {k: TRUE[k] * np.exp(OMEGA * rng.standard_normal()) for k in ("circ0", "slope")}
        sol = model.solve((0, 504), times=24 * DAYS, params=own,
                          regimen=tq.bolus("Ac", dose, times=[0]), rtol=1e-8, atol=1e-10)
        anc = np.asarray(sol["ANC"]) * np.exp(0.15 * rng.standard_normal(len(DAYS)))
        rows.append({"ID": i, "TIME": 0.0, "AMT": dose, "EVID": 1, "DV": np.nan})
        rows += [{"ID": i, "TIME": 24.0 * d, "AMT": np.nan, "EVID": 0, "DV": v} for d, v in zip(DAYS, anc, strict=True)]
    data = pd.DataFrame(rows)

    p = tf.Problem(model, name="friberg")
    p.data(data)
    p.unit("ID")
    p.doses(into="Ac")
    p.observe("ANC", "DV")
    p.estimate("circ0", tf.lognormal(median=4, sdlog=0.5))
    p.estimate("slope", tf.lognormal(median=0.3, sdlog=0.5))
    p.estimate("MTT", 80)
    p.estimate("gamma", 0.25)
    p.error("ANC", tf.proportional(0.2))
    p.method("laplace")
    p.options(seed=1, progress="none")
    return model, data, p


def main():
    model, data, p = problem()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        r = tf.fit(p)
        seconds = time.perf_counter() - t0
    truth = {"circ0.median": TRUE["circ0"], "slope.median": TRUE["slope"], "MTT": TRUE["MTT"],
             "gamma": TRUE["gamma"], "circ0.sdlog": OMEGA, "slope.sdlog": OMEGA, "ANC.proportional": 0.15}
    t = r.table().set_index("quantity")
    est = pd.DataFrame({"quantity": list(truth), "true": list(truth.values()),
                        "estimate": t.loc[list(truth), "estimate"].to_numpy(),
                        "lo": t.loc[list(truth), "CI 2.5 %"].to_numpy(),
                        "hi": t.loc[list(truth), "CI 97.5 %"].to_numpy()})
    RESULTS.mkdir(parents=True, exist_ok=True)
    est.to_csv(RESULTS / "friberg_estimates.csv", index=False)
    data.to_csv(RESULTS / "friberg_data.csv", index=False)
    (RESULTS / "friberg_time.json").write_text(json.dumps(
        {"m2ll": float(r.m2ll), "status": r.status, "seconds": seconds,
         "load_at_end": os.getloadavg()[0], "cpus": os.cpu_count()}, indent=1))
    # typical patient at each dose, at the estimates
    tt = np.linspace(0, 504, 253)
    fixed = {"circ0": r.estimates["circ0.median"], "slope": r.estimates["slope.median"],
             "MTT": r.estimates["MTT"], "gamma": r.estimates["gamma"]}
    typ = {"t": tt}
    for dose in (60, 90, 120):
        typ[f"dose_{dose}"] = np.asarray(model.solve((0, 504), times=tt, params=fixed,
                                         regimen=tq.bolus("Ac", dose, times=[0]), rtol=1e-8)["ANC"])
    pd.DataFrame(typ).to_csv(RESULTS / "friberg_typical.csv", index=False)
    print(f"-2LL {r.m2ll:.2f}, {r.status}")
    print(est.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
