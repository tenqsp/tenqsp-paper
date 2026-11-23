# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Curved geometry against a closed form: steady diffusion with first-order uptake (S5, R5).

In a sphere or an infinite cylinder of radius 1 with the concentration held at 1 on the surface,
d_t(c) = div(grad c) - phi^2 c reaches a steady state whose mean over the volume, the
effectiveness factor, is known in closed form:

    sphere:   eta = 3 (phi coth(phi) - 1) / phi^2
    cylinder: eta = 2 I1(phi) / (phi I0(phi))

The model is run to steady state on 10, 20, 40 and 80 cells for phi = 1, 3 and 10, and the mean
(int_r(c) / int_r(1), so the geometry's cell volumes are those TenQSP computes) is compared with
the closed form. Writes results/geometry/effectiveness.csv with the error and the observed order.

    python checks/geometry_effectiveness.py      # seconds
"""

from pathlib import Path

import numpy as np
import tenqsp as tq
from scipy.special import i0, i1

ROOT = Path(__file__).resolve().parent.parent
CELLS = (10, 20, 40, 80)
PHIS = (1.0, 3.0, 10.0)


def exact(geometry, phi):
    if geometry == "spherical":
        return 3 * (phi / np.tanh(phi) - 1) / phi**2
    return 2 * i1(phi) / (phi * i0(phi))


def computed(geometry, phi, n):
    m = tq.Model.from_text(f"""
model effectiveness
axis r in [0, 1] n={n} geometry={geometry}
param phi={phi}, D=1
field c(r)
field one(r)
d_t(c) = d_r(D*d_r(c)) - phi^2*c
    at r=1: c = 1
d_t(one) = 0*one
one(0) = 1
c(0) = 1
observable eta = int_r(c)/int_r(one)
""")
    t_end = 20.0  # slowest mode decays as exp(-(phi^2 + 9.87) t) or faster
    sol = m.solve((0, t_end), times=[t_end], rtol=1e-12, atol=1e-14)
    return float(np.asarray(sol["eta"])[-1])


if __name__ == "__main__":
    rows = ["geometry,phi,cells,eta,exact,relative_error,observed_order,tenqsp"]
    for geometry in ("spherical", "cylindrical"):
        for phi in PHIS:
            e, prev = exact(geometry, phi), None
            for n in CELLS:
                v = computed(geometry, phi, n)
                err = abs(v - e) / e
                order = "" if prev is None else f"{np.log2(prev / err):.2f}"
                rows.append(f"{geometry},{phi:g},{n},{v:.10g},{e:.10g},{err:.3e},{order},{tq.__version__}")
                print(rows[-1], flush=True)
                prev = err
    out = ROOT / "results" / "geometry"
    out.mkdir(parents=True, exist_ok=True)
    (out / "effectiveness.csv").write_text("\n".join(rows) + "\n")
