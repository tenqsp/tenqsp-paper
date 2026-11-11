# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Reproduce Lorz et al. 2015 (Bull Math Biol 77:1-22) with TenQSP, then run the true sphere.

The published figures correspond to a flat (one-dimensional) Laplacian: run on a planar axis,
the model gives the paper's nutrient and drug levels at the centre (its Figure 2) and the
resistance level selected at each depth (its Figure 1). At each depth, the cells concentrate
at a root of the net growth rate R(x) = 0 (the paper's eqs 3.3 and 3.4). The same is then
run with geometry=spherical. About a minute in all.
"""

from pathlib import Path

import numpy as np
import tenqsp as tq

HERE = Path(__file__).resolve().parent
TEXT = (HERE.parent / "models" / "spheroid_lorz2015.tq").read_text()
NX = 100
XC = (np.arange(NX) + 0.5) / NX
SOLVER = {"rtol": 1e-6, "atol": 1e-30}  # the tails of n near x = 0 (~e^-50) need a tiny atol


def run(geometry, C1, T=700.0):
    m = tq.Model.from_text(TEXT.replace("geometry=spherical", f"geometry={geometry}"))
    sol = m.solve((0, T), times=[T], params={"C1": C1}, **SOLVER)
    s, c1, c2 = (np.asarray(sol[k])[-1] for k in ("s", "c1", "c2"))
    n = np.asarray(sol["n"])[-1]
    rho = n.sum(axis=1) / NX
    peak = XC[np.argmax(n, axis=1)]
    roots = []  # R(x) = (1.1 - x) S - (3 + 1.5 (1-x)^2) rho - (0.01 + (1-x)^2) c1 = 0
    for i in range(len(s)):
        S = s[i] / (1 + 10 * c2[i])
        a, b = -1.5 * rho[i] - c1[i], -S + 3 * rho[i] + 2 * c1[i]
        c = 1.1 * S - 4.5 * rho[i] - 1.01 * c1[i]
        r = np.real(np.roots([a, b, c]))
        r = r[(r > -0.05) & (r < 1.05)]
        roots.append(r[np.argmin(np.abs(r - peak[i]))])
    return s, c1, peak, np.array(roots)


def check(what, ok):
    print(f"{'OK ' if ok else 'XX '} {what}")


s, _, peak, root = run("planar", 0.0)
print("planar, no drug: s(0) =", round(s[0], 4), " peaks at r = 0, 0.5, 1:", peak[[0, 20, -1]])
check("s at the centre = paper's Figure 2 (about 0.063)", abs(s[0] - 0.063) < 0.003)
check("peaks = paper's Figure 1 (about 0.43 at the centre, 0.22 at the surface)",
      abs(peak[0] - 0.43) < 0.02 and abs(peak[-1] - 0.22) < 0.02)
check("peaks on the root of R = 0 (eq 3.3), within 0.01", np.max(np.abs(peak - root)) < 0.01)

_, c1, peak, root = run("planar", 1.0)
print("planar, cytotoxic drug: c1(0) =", round(c1[0], 4))
check("c1 at the centre = paper's Figure 2 (about 0.21)", abs(c1[0] - 0.21) < 0.01)

s_sph, _, peak_sph, root_sph = run("spherical", 0.0)
print("sphere, no drug: s(0) =", round(s_sph[0], 4), " peaks:", peak_sph[[0, 20, -1]])
check("in a sphere the centre gets more nutrient", s_sph[0] > 2 * s[0])
check("sphere: peaks on the root of R = 0, within 0.01", np.max(np.abs(peak_sph - root_sph)) < 0.01)
