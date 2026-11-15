# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The influenza model of Figure 2 against the published model it builds on (Baccam et al.
2006, J Virol 80:7590): with one eclipse cell, models/influenza_eclipse.tq is Baccam's
equations 5-8, and R0 = p beta T0 / (c delta) (their equation 4). Under 1 min.

    python checks/baccam2006_reproduction.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "figures"))
import eclipse as ec

K = 1 / ec.TAU  # 4 per day: Baccam's k, 1/k = 6 h


def baccam(t, y):
    """Equations 5-8: T, I1 (eclipse), I2 (productive), V."""
    T, I1, I2, V = y
    return [-ec.BETA * T * V, ec.BETA * T * V - K * I1, K * I1 - ec.DELTA * I2, ec.P * I2 - ec.C * V]


t = np.linspace(0, 7, 701)
ref = solve_ivp(baccam, (0, 7), [ec.T0, 0, 0, ec.V0], t_eval=t, method="BDF", rtol=1e-11, atol=1e-14)
sol = ec.eclipse_model(1).solve((0, 7), times=t, rtol=1e-11, atol=1e-14)
V, Vref = np.asarray(sol["V"]), ref.y[3]
rel = np.max(np.abs(V - Vref) / Vref)
print(f"viral load, TenQSP (1 eclipse cell) against equations 5-8: largest relative difference {rel:.1e}")
assert rel < 1e-6

R0 = ec.P * ec.BETA * ec.T0 / (ec.C * ec.DELTA)
print(f"R0 from the geometric-mean parameters of Table 3: {R0:.1f} (Table 3: 21.5, the geometric "
      "mean of the six patients' R0)")
assert abs(R0 - 21.5) < 0.5
i = np.argmax(V)
print(f"peak {V[i]:.2g} TCID50/mL at day {t[i]:.2f} (Figure 1: peaks of about 1e5-1e7 on days 2-3)")
assert 2 <= t[i] <= 3
