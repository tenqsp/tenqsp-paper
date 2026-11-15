# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Shared set-up of Figure 5 A-C: transit absorption with a delay of known dispersion, studies
simulated from it, and fits with fixed transit chains (tenqsp-fit).

The truth: the drug in transit moves along an axis with dispersion (`using sharp` on 200
cells), so the transit time has mean MTT and CV sqrt(2 disp) = 0.20, the CV of an Erlang
chain of 25, but a longer right tail than any chain (skewness 0.60; an Erlang chain with
the same CV has 0.40). No chain matches it exactly. A study: 12 subjects, 11 samples from 15 min to 24 h, combined error (0.02 mg/L
plus 15 %), samples below the limit of quantification (0.05 mg/L) reported as such and
fitted by censoring (M3). Each study is fitted with chains of N compartments (the erlang
scheme on N cells) for several N: the chain length is the structure being estimated.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tenqsp as tq
import tenqsp_fit as tf

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "fig5"
TRUE = {"MTT": 1.5, "ka": 3.0, "CL": 3.0, "V": 30.0}
DISP = 0.02  # transit-time CV sqrt(2 * 0.02) = 0.20
TIMES = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 8, 24]
SUBJECTS, CV, SD, LLOQ = 12, 0.15, 0.02, 0.05
CHAINS = (3, 5, 10, 15, 20, 25, 30, 40, 60)


def transit_model(n, dispersion=False):
    """Transit absorption, one-compartment disposition, a dose of 100 mg. On n cells: a chain
    of n transit compartments (erlang), or with dispersion (sharp) the model of the truth."""
    eq = "d_t(u) = -d_z(u/MTT)"
    if dispersion:
        eq = "d_t(u) = -d_z(u/MTT) + d_z(disp/MTT*d_z(u))    using sharp"
    return tq.Model.from_text(f"""
model transit
axis z in [0, 1] n={n} kind=one_way
param MTT=1.5, ka=3, CL=3, V=30, D=100{f", disp={DISP}" if dispersion else ""}
field u(z)
{eq}
    at z=1: outflow
d_t(Aa) = u(z=1)/MTT - ka*Aa
d_t(Ac) = ka*Aa - CL/V*Ac
u(0) = D*point(z, at={0.5 / n})
observable conc = Ac/V
""")


def true_profile(times=TIMES):
    return np.asarray(transit_model(200, True).solve((0, 24), times=times, rtol=1e-10, atol=1e-12)["conc"])


def study(seed, true=None):
    """One simulated study: combined error, values below LLOQ flagged (BLQ = 1)."""
    true = true_profile() if true is None else true
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(1, SUBJECTS + 1):
        for t, c in zip(TIMES, true, strict=True):
            y = c * (1 + CV * rng.standard_normal()) + SD * rng.standard_normal()
            rows.append({"ID": i, "TIME": t, "DV": max(y, LLOQ), "BLQ": int(y < LLOQ)})
    return pd.DataFrame(rows)


def problem(n, data, starts=4):
    p = tf.Problem(transit_model(n), name=f"chain{n}")
    p.data(data)
    p.unit("ID")
    p.observe("conc", "DV")
    p.estimate("MTT", 1.0)
    p.estimate("ka", 2.0, bounds=(0.3, 100))
    p.estimate("CL", 2.0)
    p.estimate("V", 20.0)
    p.fix("D", 100)
    p.error("conc", tf.combined(0.05, 0.2))
    p.censor("conc", below=LLOQ, where="BLQ == 1")
    p.method("optimize")
    p.options(starts=starts, seed=1, checks="quick", progress="none")
    return p
