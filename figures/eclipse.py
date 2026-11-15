# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Shared set-up of Figure 2: the transit and influenza-eclipse models, their parameters, and
the theory they are checked against. Used by fig2_runs.py, fig2_transit.py and the checks.

The early growth rate r of the infection solves the characteristic equation
(r + c)(r + delta) = p beta T0 L(r), where L is the Laplace transform of the eclipse duration:
(1 + r tau/N)^-N for an Erlang(N) eclipse with mean tau, exp(-r tau) for a fixed delay.
R0 = p beta T0 / (c delta) does not depend on L, so neither does the threshold r = 0.
"""

import re
from pathlib import Path

import numpy as np
import tenqsp as tq
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results" / "fig2"

# Baccam et al. 2006, Table 3 (geometric means), and T0 fixed in the paper
BETA, P, C, DELTA, TAU, T0, V0 = 3.2e-5, 4.6e-2, 5.2, 5.2, 0.25, 4e8, 7.5e-2
T_STAR = C * DELTA / (P * BETA)  # the threshold: R0 = 1 at T0 = T_STAR
SOLVER = {"rtol": 1e-8, "atol": 1e-14}
SHARP_CELLS = 200  # the fixed delay: the sharp scheme on 200 cells of 1.8 min


def eclipse_model(n, sharp=False):
    """The influenza model with the eclipse on n cells (erlang: an Erlang(n) eclipse)."""
    text = (MODELS / "influenza_eclipse.tq").read_text()
    text = re.sub(r"(axis a in \[0, 0.25\] n=)\d+", rf"\g<1>{n}", text)
    if sharp:
        text = text.replace("d_t(e) = -d_a(e)\n", "d_t(e) = -d_a(e)    using sharp\n")
    return tq.Model.from_text(text)


def laplace(n):
    """Laplace transform of the eclipse duration: Erlang(n), or a fixed delay (n=None)."""
    if n is None:
        return lambda r: np.exp(-r * TAU)
    return lambda r: (1 + r * TAU / n) ** -n


def growth_rate(n, t0=T0):
    """The root of the characteristic equation: the early growth rate (1/day)."""
    lap = laplace(n)

    def f(r):
        return (r + C) * (r + DELTA) - P * BETA * t0 * lap(r)

    if f(0.0) < 0:
        return brentq(f, 0.0, 100.0, xtol=1e-14)
    lower = -min(C, DELTA, np.inf if n is None else n / TAU)
    return brentq(f, lower * (1 - 1e-9), 0.0, xtol=1e-14)


def leading_eigenvalue(model, t0=T0):
    """The largest real part of the eigenvalues of TenQSP's Jacobian at the infection-free
    state with t0 target cells (the T direction, a zero eigenvalue, is left out)."""
    cm = model.compile()
    y = np.zeros(cm.n_states)
    iT = cm.state_index("T")
    y[iT] = t0
    J = np.delete(np.delete(np.asarray(cm.jacobian(0.0, y)), iT, 0), iT, 1)
    return float(np.max(np.linalg.eigvals(J).real))


def transit_models(n):
    """Savic's transit model with n cells on the axis, and the same written as a chain of n
    compartments (A0 ... A{n-1}) by hand."""
    text = (MODELS / "savic_transit.tq").read_text()
    text = re.sub(r"(axis z in \[0, 1\] n=)\d+", rf"\g<1>{n}", text)
    text = text.replace("point(z, at=0.125)", f"point(z, at={0.5 / n})")
    lines = ["model transit_chain", "param MTT=2, ka=1.2, CL=3, V=10", f"ktr = {n}/MTT"]
    lines.append("d_t(A0) = -ktr*A0")
    lines += [f"d_t(A{i}) = ktr*A{i - 1} - ktr*A{i}" for i in range(1, n)]
    lines += [f"d_t(Aa) = ktr*A{n - 1} - ka*Aa", "d_t(Ac) = ka*Aa - CL/V*Ac"]
    lines += ["A0(0) = 100", "observable conc = Ac/V"]
    return tq.Model.from_text(text), tq.Model.from_text("\n".join(lines))
