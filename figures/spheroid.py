# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Shared set-up of the spheroid case study (Figure 7): the models, the grids, the dosing
schedules and the solver settings. Used by fig7_runs.py, fig7_gradients.py and fig7_spheroid.py.

Two modes: "full" (the paper's grid, 40 depth cells x 100 resistance levels) and "quick"
(20 x 50; for trying the scripts: within 9 % of the full grid for a saturable kill, up to 24 % for
the linear kill, see Supplement S8).
"""

import re
from pathlib import Path

import numpy as np
import tenqsp as tq

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
GRIDS = {"full": (40, 100), "quick": (20, 50)}
T = 1000.0  # length of the course
KE, CAVG = 0.7, 2.0  # plasma elimination and the average plasma level of every schedule
SCHEDULES = {"inf": "infusion", "7": "bolus every 7", "21": "bolus every 21"}
# The tails of n near x = 0 (~e^-50) are amplified by selection: they need a tiny atol.
SOLVER = {"rtol": 1e-6, "atol": 1e-25}


def model(name="spheroid_pk", mode="full", extra=""):
    """A case-study model on the grid of ``mode``, with the lines ``extra`` appended."""
    text = (MODELS / f"{name}.tq").read_text() + extra
    nr, nx = GRIDS[mode]
    text = re.sub(r"(axis r in \[0, 1\] n=)\d+", rf"\g<1>{nr}", text)
    text = re.sub(r"(axis x in \[0, 1\] n=)\d+", rf"\g<1>{nx}", text)
    return tq.Model.from_text(text)


def dose_times(schedule):
    return [] if schedule == "inf" else list(np.arange(0.0, T, float(schedule)))


def regimen(schedule):
    """The same average plasma level CAVG for every schedule: an infusion at CAVG*KE, or a
    bolus of CAVG*KE*tau every tau."""
    if schedule == "inf":
        return tq.infusion("P1", rate=CAVG * KE, duration=T, times=[0.0])
    tau = float(schedule)
    return tq.bolus("P1", CAVG * KE * tau, times=dose_times(schedule))


def results_dir(mode):
    d = RESULTS / f"fig7_{mode}"
    d.mkdir(parents=True, exist_ok=True)
    return d
