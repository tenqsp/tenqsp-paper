# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Figure 2, the computations: writes results/fig2/*.npz (under 1 min).

    python figures/fig2_runs.py
"""

import eclipse as ec
import numpy as np

NS = (1, 2, 3, 5, 10, 20, 40)  # eclipse compartments shown in panel C
SHOWN = (1, 5, 40)  # and in panels B, D and E


def transit():
    """Panel A: the transit model on the axis against the hand-written chain."""
    t = np.linspace(0, 12, 241)
    out = {"t": t}
    for n in (4, 10):
        axis, chain = ec.transit_models(n)
        out[f"axis_{n}"] = np.asarray(axis.solve((0, 12), times=t, rtol=1e-11, atol=1e-13)["conc"])
        out[f"chain_{n}"] = np.asarray(chain.solve((0, 12), times=t, rtol=1e-11, atol=1e-13)["conc"])
        print(f"transit, {n} cells: largest difference {np.max(np.abs(out[f'axis_{n}'] - out[f'chain_{n}'])):.1e} mg/L")
    np.savez_compressed(ec.RESULTS / "transit.npz", **out)


def slope(model):
    """The early growth rate from a simulation: started from a tiny inoculum, so that the
    start-up transient is over and the target cells are untouched (T/T0 > 0.999) while
    the viral load grows from 1e-3 to 1e2."""
    t = np.linspace(0, 6, 2401)
    v = np.asarray(model.solve((0, 6), times=t, params={"V0": 1e-8}, **ec.SOLVER)["V"])
    w = (v >= 1e-3) & (v <= 1e2)
    return np.polyfit(t[w], np.log(v[w]), 1)[0]


def eclipse():
    """Panels C-E: growth rates (theory, Jacobian, simulation) and viral load curves."""
    t = np.linspace(0, 6, 1201)
    out = {"t": t, "ns": np.array(NS)}
    out["root"] = np.array([ec.growth_rate(n) for n in NS])
    out["eig"] = np.array([ec.leading_eigenvalue(ec.eclipse_model(n)) for n in NS])
    out["sim"] = np.array([slope(ec.eclipse_model(n)) for n in NS])
    out["root_delay"] = ec.growth_rate(None)
    out["sim_delay"] = slope(ec.eclipse_model(ec.SHARP_CELLS, sharp=True))
    sims = {n: ec.eclipse_model(n).solve((0, 6), times=t, **ec.SOLVER) for n in SHOWN}
    sims["delay"] = ec.eclipse_model(ec.SHARP_CELLS, sharp=True).solve((0, 6), times=t, **ec.SOLVER)
    for k in (*SHOWN, "delay"):
        out[f"V_{k}"] = np.asarray(sims[k]["V"])
        out[f"T_{k}"] = np.asarray(sims[k]["T"])
    # panel D: growth rate against the number of target cells, through the threshold
    t0s = ec.T_STAR * np.geomspace(0.3, 30, 61)
    out["t0s"] = t0s
    for k in (*SHOWN, None):
        out[f"r_vs_T0_{k or 'delay'}"] = np.array([ec.growth_rate(k, x) for x in t0s])
    t0_marks = ec.T_STAR * np.array([0.5, 1.0, 2.0, 5.0, 21.75])
    out["t0_marks"] = t0_marks
    for k in SHOWN:
        m = ec.eclipse_model(k)
        out[f"eig_vs_T0_{k}"] = np.array([ec.leading_eigenvalue(m, x) for x in t0_marks])
    for i, n in enumerate(NS):
        print(f"N = {n:2d}: root {out['root'][i]:.6f}, Jacobian {out['eig'][i]:.6f}, "
              f"simulation {out['sim'][i]:.6f} per day")
    print(f"delay: root {out['root_delay']:.6f}, simulation (sharp, {ec.SHARP_CELLS} cells) {out['sim_delay']:.6f}")
    np.savez_compressed(ec.RESULTS / "eclipse.npz", **out)


if __name__ == "__main__":
    ec.RESULTS.mkdir(parents=True, exist_ok=True)
    transit()
    eclipse()
