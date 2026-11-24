# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Friberg's model fitted by tenqsp-fit (figures/fig3_friberg.py) and by NONMEM 7.6 on the same
simulated study, from the same starts, both by FOCE-I (NONMEM: METHOD=1 INTER). Table S1.

    python checks/nonmem/friberg_nonmem.py data    # writes checks/nonmem/friberg_nm.csv
    nmfe76 friberg.ctl friberg.lst                 # in checks/nonmem (needs a NONMEM licence)
    python checks/nonmem/friberg_nonmem.py         # Table S1 from results/

The table reads tenqsp-fit's estimates from results/fig3/friberg_estimates.csv and NONMEM's
from results/nonmem/friberg.ext (copy friberg.ext there after a NONMEM run). NONMEM's
objective leaves out n log(2 pi); it is added back to compare -2LL. Standard errors: NONMEM's
from the R matrix ($COVARIANCE MATRIX=R, the matrix tenqsp-fit uses), on the scale of each
THETA, and for OMEGA on the variance, carried to the SD; tenqsp-fit's from its 95% interval,
which it computes on the log scale: SE = estimate x (log hi - log lo) / (2 x 1.96).
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "results" / "fig3" / "friberg_data.csv"
EXT = ROOT / "results" / "nonmem" / "friberg.ext"

# tenqsp-fit quantity -> (label, NONMEM column of the .ext file, is an OMEGA variance)
ROWS = [
    ("circ0.median", "CIRC0, typical (10^9/L)", "THETA1", False),
    ("slope.median", "Slope, typical (L/mg)", "THETA2", False),
    ("MTT", "MTT (h)", "THETA3", False),
    ("gamma", "gamma", "THETA4", False),
    ("circ0.sdlog", "omega CIRC0 (SD, log scale)", "OMEGA(1,1)", True),
    ("slope.sdlog", "omega slope (SD, log scale)", "OMEGA(2,2)", True),
    ("ANC.proportional", "Proportional error (SD)", "THETA5", False),
]


def write_data():
    """The study in NONMEM's layout: doses into the drug compartment (CMT 6), counts in CIRC (5)."""
    d = pd.read_csv(DATA)
    dose = d.EVID == 1
    out = pd.DataFrame({
        "ID": d.ID, "TIME": d.TIME, "AMT": d.AMT.fillna(0), "EVID": d.EVID,
        "MDV": dose.astype(int), "CMT": dose.map({True: 6, False: 5}), "DV": d.DV.fillna(0),
    })
    out.to_csv(HERE / "friberg_nm.csv", index=False)
    print(f"wrote {HERE / 'friberg_nm.csv'}: {d.ID.nunique()} subjects, {(~dose).sum()} observations")


def read_ext():
    """Final estimates (ITERATION -1000000000) and standard errors (-1000000001)."""
    ext = pd.read_csv(EXT, sep=r"\s+", skiprows=1)
    est = ext[ext.ITERATION == -1000000000].iloc[0]
    se = ext[ext.ITERATION == -1000000001].iloc[0]
    return est, se


def table():
    tq = pd.read_csv(ROOT / "results" / "fig3" / "friberg_estimates.csv").set_index("quantity")
    est, se = read_ext()
    n_obs = int((pd.read_csv(DATA).EVID == 0).sum())
    rows = []
    for q, label, col, omega in ROWS:
        e, lo, hi = tq.loc[q, ["estimate", "lo", "hi"]]
        tq_se = e * (math.log(hi) - math.log(lo)) / (2 * 1.959964)
        nm, nm_se = est[col], se[col]
        if omega:  # variance -> SD, and its SE by the delta method
            nm, nm_se = math.sqrt(nm), nm_se / (2 * math.sqrt(nm))
        rows.append({"Parameter": label, "True": tq.loc[q, "true"], "tenqsp-fit": e, "NONMEM": nm,
                     "Difference (%)": 100 * (e - nm) / nm, "SE tenqsp-fit": tq_se, "SE NONMEM": nm_se})
    out = pd.DataFrame(rows)
    m2ll_nm = est["OBJ"] + n_obs * math.log(2 * math.pi)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\n-2LL: NONMEM {m2ll_nm:.3f} (objective {est['OBJ']:.4f} + {n_obs} log(2 pi))", end="")
    fit = json.loads((ROOT / "results" / "fig3" / "friberg_time.json").read_text())
    print(f", tenqsp-fit {fit['m2ll']:.3f}" if "m2ll" in fit else " (tenqsp-fit's: rerun fig3_friberg.py)")
    (ROOT / "out").mkdir(exist_ok=True)
    out.to_csv(ROOT / "out" / "tableS1_nonmem.csv", index=False)


if __name__ == "__main__":
    if sys.argv[1:] == ["data"]:
        write_data()
    else:
        table()
