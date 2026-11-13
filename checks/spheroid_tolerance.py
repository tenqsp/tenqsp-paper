# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Check that the Figure 7 results do not depend on the solver's absolute tolerance.

The relapse grows from cells that start very small (2e-23 at full resistance), so atol must
be far below them. Each run is repeated at atol 1e-15, 1e-20 and 1e-28 and compared with the
paper's setting (1e-25): linear kill (the tumour falls to about 1e-11, the hardest case) and
EC50 = 4, under an infusion and a bolus every 21. A coarse atol (1e-9) is shown for contrast
and not checked: with linear kill it gives a negative tumour burden.

    python checks/spheroid_tolerance.py           # quick grid (20 x 50): under a minute
    python checks/spheroid_tolerance.py --full    # the paper's grid
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "figures"))
import spheroid as sp

CASES = (("linear kill", 0.0, "inf"), ("EC50 4", 0.25, "inf"), ("EC50 4", 0.25, "21"))
ATOLS = (1e-9, 1e-15, 1e-20, 1e-25, 1e-28)
REFERENCE = sp.SOLVER["atol"]


def one(job):
    mode, isat, schedule, atol = job
    sol = sp.model("spheroid_pk", mode).solve(
        (0, sp.T), times=[0, sp.T], regimen=sp.regimen(schedule), params={"isat": isat},
        rtol=sp.SOLVER["rtol"], atol=atol)
    return float(np.asarray(sol["total"])[-1]), float(np.asarray(sol["mean_x"])[-1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    mode = "full" if ap.parse_args().full else "quick"
    jobs = [(mode, isat, s, a) for _, isat, s in CASES for a in ATOLS]
    with ProcessPoolExecutor(8) as ex:
        v = dict(zip(((i, s, a) for _, i, s, a in jobs), ex.map(one, jobs), strict=True))
    ok = True
    for name, isat, s in CASES:
        ref_b, ref_x = v[(isat, s, REFERENCE)]
        print(f"{name}, {sp.SCHEDULES[s]}: burden at the end {ref_b:.5g}, resistance {ref_x:.5f}"
              f" (atol {REFERENCE:g})")
        for a in ATOLS:
            if a == REFERENCE:
                continue
            b, x = v[(isat, s, a)]
            checked = a <= 1e-15
            good = abs(b - ref_b) <= 1e-3 * abs(ref_b) and abs(x - ref_x) <= 1e-4
            if checked:
                ok &= good
            mark = ("OK " if good else "XX ") if checked else "   "
            print(f"  {mark}atol {a:g}: burden {b: .5g}, resistance {x:.5f}"
                  + ("" if checked else "  (for contrast, not checked)"))
    sys.exit(0 if ok else 1)
