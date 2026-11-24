# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Check that the Figure 6 results do not depend on the solver's absolute tolerance.

The relapse grows from cells that start very small (2e-23 at full resistance). Each run is
repeated at atol 1e-15, 1e-20 and 1e-28 and at rtol 1e-8, and compared with the paper's setting
(rtol 1e-6, atol 1e-25): linear kill (the tumor falls to about 1e-11, the hardest case) and
EC50 = 4, under an infusion and a bolus every 21. A coarse atol (1e-9) is shown for contrast
and not checked: with linear kill it gives a negative tumor burden. Writes
results/spheroid_tolerance/tolerance_<mode>.csv.

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
TIGHT_RTOL = 1e-8


def one(job):
    mode, isat, schedule, atol, rtol = job
    sol = sp.model("spheroid_pk", mode).solve(
        (0, sp.T), times=[0, sp.T], regimen=sp.regimen(schedule), params={"isat": isat},
        rtol=rtol, atol=atol)
    return float(np.asarray(sol["total"])[-1]), float(np.asarray(sol["mean_x"])[-1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    mode = "full" if ap.parse_args().full else "quick"
    R = sp.SOLVER["rtol"]
    jobs = [(mode, isat, s, a, R) for _, isat, s in CASES for a in ATOLS]
    jobs += [(mode, isat, s, REFERENCE, TIGHT_RTOL) for _, isat, s in CASES]
    with ProcessPoolExecutor(len(jobs)) as ex:
        v = dict(zip(((i, s, a, r) for _, i, s, a, r in jobs), ex.map(one, jobs), strict=True))
    rows = ["kill,schedule,rtol,atol,burden_end,resistance_end,relative_difference_burden"]
    for name, isat, s in CASES:
        ref_b = v[(isat, s, REFERENCE, R)][0]
        for (i, s2, a, r), (b, x) in v.items():
            if i == isat and s2 == s:
                rows.append(f"{name},{s},{r:g},{a:g},{b:.8g},{x:.8f},{abs(b - ref_b) / abs(ref_b):.3g}")
    out = Path(__file__).resolve().parent.parent / "results" / "spheroid_tolerance"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"tolerance_{mode}.csv").write_text("\n".join(rows) + f"\n# tenqsp {sp.tq.__version__}\n")
    v = {(i, s, a) if r == R else (i, s, "rtol"): val for (i, s, a, r), val in v.items()}
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
        b, x = v[(isat, s, "rtol")]
        good = abs(b - ref_b) <= 1e-3 * abs(ref_b) and abs(x - ref_x) <= 1e-4
        ok &= good
        print(f"  {'OK ' if good else 'XX '}rtol {TIGHT_RTOL:g}: burden {b: .5g}, resistance {x:.5f}")
    sys.exit(0 if ok else 1)
