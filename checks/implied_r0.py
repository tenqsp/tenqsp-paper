# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The R0 implied by an observed early growth rate, for each eclipse shape (R1).

Linearized at the infection-free state, the influenza model of R1 grows at the rate r that solves
(r + c)(r + delta) = p beta T0 L(r), with L the Laplace transform of the eclipse duration
(Supplement S6). Since R0 = p beta T0 / (c delta), a growth rate r implies
R0 = (r + c)(r + delta) / (c delta L(r)). For the growth rate of the published model (one eclipse
compartment, at Baccam's estimates), this gives R0 under each assumed shape and the critical
antiviral efficacy 1 - 1/R0. c, delta and the mean eclipse duration are held at Baccam's values.
Writes results/fig2/implied_r0.csv.

    python checks/implied_r0.py      # seconds
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
C = DELTA = 5.2  # day^-1 (Baccam et al. 2006)
TAU = 0.25  # day, the mean eclipse duration
SHAPES = {"1": 1, "2": 2, "5": 5, "10": 10, "40": 40, "fixed": None}


def laplace(r, n):
    """Laplace transform of the eclipse duration: Erlang(n) with mean TAU, or a fixed delay."""
    return np.exp(-r * TAU) if n is None else (1 + r * TAU / n) ** (-n)


def implied_r0(r, n):
    return (r + C) * (r + DELTA) / (C * DELTA * laplace(r, n))


if __name__ == "__main__":
    e = np.load(ROOT / "results" / "fig2" / "eclipse.npz")
    r = float(np.asarray(e["root"]).ravel()[0])  # the published model's growth rate (one compartment)
    rows = ["eclipse,growth_rate_per_day,implied_R0,critical_efficacy"]
    for name, n in SHAPES.items():
        r0 = implied_r0(r, n)
        rows.append(f"{name},{r:.6g},{r0:.6g},{1 - 1 / r0:.6g}")
    out = ROOT / "results" / "fig2" / "implied_r0.csv"
    out.write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
