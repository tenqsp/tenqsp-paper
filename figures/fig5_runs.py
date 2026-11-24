# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Figure 5, the computations: writes results/fig5/*.npz and the exact checks (about 1 min).

    python figures/fig5_runs.py
"""

import json

import numpy as np
import pbpk as pb
import tenqsp as tq
from scipy.integrate import solve_ivp

DOSE, RATE = 100.0, 10.0  # mg IV bolus; mg/h infusion
PS_GRID = (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


def main():
    pb.RESULTS.mkdir(parents=True, exist_ok=True)
    m = pb.model()
    checks = {}
    iv = tq.bolus("Cven", DOSE, times=[0])
    CL_H = pb.hepatic_clearance()

    # the same model as 29 hand-written ODEs
    t = np.array([0.1, 1, 6, 24, 96.0])
    sol = m.solve((0, 96), times=t, regimen=iv, **pb.TIGHT)
    rhs, Vt = pb.by_hand()
    y0 = np.zeros(29)
    y0[28] = DOSE
    ref = solve_ivp(rhs, (0, 96), y0, t_eval=t, method="Radau", rtol=1e-12, atol=1e-14).y
    hand = ref[:27].T.reshape(-1, 9, 3) / Vt
    checks["field = 29 hand-written ODEs"] = float(np.max(np.abs(np.asarray(sol["C"]) - hand) / (hand + 1e-6)))

    # an IV dose: profiles, mass balance and AUC
    times = np.concatenate([np.linspace(0, 2, 81)[1:], np.linspace(2, 600, 300)[1:]])
    sol = m.solve((0, 600), times=times, regimen=iv, **pb.TIGHT)
    eliminated = -sol.ledger.moved["C"]["hazard: fu * CLint / Kp[organ] / V"]
    in_blood = pb.V_ART * np.asarray(sol["Cart"]) + pb.V_VEN * np.asarray(sol["Cven"])
    total = np.asarray(sol["in_tissues"]) + in_blood + eliminated
    checks["mass balance (mg, of 100)"] = float(np.max(np.abs(total - DOSE)) / DOSE)
    checks["AUC = Dose / CL_H"] = abs(float(sol["AUC"][-1]) / (DOSE / CL_H) - 1)
    np.savez_compressed(pb.RESULTS / "iv.npz", t=times, C=np.asarray(sol["C"]),
                        Cart=np.asarray(sol["Cart"]), eliminated=eliminated)

    # an infusion: steady state and tissue:plasma ratios
    infusion = tq.infusion("Cven", rate=RATE, duration=1e4, start=0)
    ss = m.solve((0, 800), times=[800.0], regimen=infusion, **pb.TIGHT)
    C_end, c_art = np.asarray(ss["C"])[-1], float(ss["Cart"][-1])
    checks["steady state Css = R / CL_H"] = abs(c_art / (RATE / CL_H) - 1)
    kp_tissue = pb.FRAC[0] + pb.FRAC[1] + pb.FRAC[2] * pb.KP
    ratio = (C_end @ pb.FRAC) / c_art
    keep = [o for o, name in enumerate(pb.ORGANS) if name != "liver"]
    checks["tissue:plasma = Kp_tissue"] = float(np.max(np.abs(ratio[keep] / kp_tissue[keep] - 1)))

    # hepatic clearance against the membranes' permeability
    simulated = []
    for ps in PS_GRID:
        s = m.solve((0, 5000), times=[5000.0], regimen=infusion, params={"PS1": ps, "PS2": ps}, **pb.TIGHT)
        simulated.append(RATE / float(s["Cart"][-1]))
        print(f"PS {ps:6.0f}: CL_H simulated {simulated[-1]:.6f}, series formula "
              f"{pb.hepatic_clearance(ps, ps):.6f} L/h", flush=True)
    formula = np.array([pb.hepatic_clearance(ps, ps) for ps in PS_GRID])
    checks["CL_H = series formula, PS 3 to 1000"] = float(np.max(np.abs(np.array(simulated) / formula - 1)))
    np.savez_compressed(pb.RESULTS / "clearance.npz", ps=np.array(PS_GRID), simulated=np.array(simulated),
                        ratio=ratio, kp_tissue=kp_tissue)

    for k, v in checks.items():
        print(f"{k:40s} {v:.1e}")
    (pb.RESULTS / "checks.json").write_text(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
