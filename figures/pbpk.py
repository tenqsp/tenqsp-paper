# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Shared set-up of Figure 6 (whole-body PBPK): the model, its values, the same model written
as 29 ODEs by hand, and the closed forms it is checked against."""

from pathlib import Path

import numpy as np
import tenqsp as tq

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "fig6"
ORGANS = ["lung", "brain", "heart", "muscle", "adipose", "kidney", "gut", "spleen", "liver"]
SPACES = ["vascular", "interstitial", "cellular"]
VORG = np.array([1.2, 1.4, 0.33, 29, 15, 0.31, 1.2, 0.19, 1.8])
FRAC = np.array([0.05, 0.15, 0.8])
KP = np.array([1, 0.5, 2, 1.5, 5, 2, 2, 1.5, 3.0])
Q_ART = np.array([0, 45, 15, 60, 15, 70, 65, 10, 20.0])
Q_VEN = np.array([0, 45, 15, 60, 15, 70, 0, 0, 95.0])
Q_H, Q_CO, V_ART, V_VEN = 95.0, 300.0, 1.7, 3.4
DEFAULTS = {"PS1": 50.0, "PS2": 10.0, "CLint": 30.0, "fu": 0.5}
TIGHT = {"rtol": 1e-10, "atol": 1e-12}


def model():
    return tq.Model.from_text((ROOT / "models" / "pbpk_wholebody.tq").read_text())


def hepatic_clearance(PS1=50.0, PS2=10.0, CLint=30.0, fu=0.5):
    """Membranes and enzymes in series, then the well-stirred liver with G for fu CLint."""
    G = 1.0 / (1.0 / PS1 + 1.0 / PS2 + 1.0 / (fu * CLint))
    return Q_H * G / (Q_H + G)


def well_stirred(CLint=30.0, fu=0.5):
    return Q_H * fu * CLint / (Q_H + fu * CLint)


def by_hand(PS1=50.0, PS2=10.0, CLint=30.0, fu=0.5):
    """The model as 29 ODEs in amounts (organ x space, then arterial and venous blood), as
    one would write it for solve_ivp or NONMEM's $DES."""
    Vt = np.outer(VORG, FRAC)
    lung, gut, spleen, liver = 0, 6, 7, 8

    def rhs(_, y):
        c = y[:27].reshape(9, 3) / Vt
        c_art, c_ven = y[27] / V_ART, y[28] / V_VEN
        dA = np.zeros((9, 3))
        to_isf = PS1 * (c[:, 0] - c[:, 1])
        to_cells = PS2 * (c[:, 1] - c[:, 2] / KP)
        dA[:, 0] += Q_ART * c_art - Q_VEN * c[:, 0] - to_isf
        dA[:, 1] += to_isf - to_cells
        dA[:, 2] += to_cells
        for organ, q in ((gut, 65.0), (spleen, 10.0)):
            dA[organ, 0] -= q * c[organ, 0]
            dA[liver, 0] += q * c[organ, 0]
        dA[lung, 0] += Q_CO * (c_ven - c[lung, 0])
        dA[liver, 2] -= fu * CLint * c[liver, 2] / KP[liver]
        d_art = Q_CO * c[lung, 0] - Q_ART.sum() * c_art
        d_ven = (Q_VEN * c[:, 0]).sum() - Q_CO * c_ven
        return np.concatenate([dA.ravel(), [d_art, d_ven]])

    return rhs, Vt
