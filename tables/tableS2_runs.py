# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Table S2 and Figure 4, the computations: runs TenQSP's 12 tutorials and writes the numbers
their checks measure to results/tableS2/<NN>.json (about 5 min, one tutorial per process).

    python tables/tableS2_runs.py                       # tutorials of the installed version
    python tables/tableS2_runs.py --tutorials DIR       # tutorials in a local folder
    python tables/tableS2_runs.py --only 01 05          # some of them

Without --tutorials, the source distribution of the installed TenQSP version is downloaded
from PyPI (it contains docs/tutorials) into a cache folder. Each tutorial is run as a script
from its own folder; its checks (asserts) run as part of it. The numbers are then read from
the tutorial's variables. Where a tutorial measures a quantity inside a loop, so that only the
last value is left, the other values are recomputed here from the models and functions the
tutorial defined.
"""

import argparse
import contextlib
import io
import json
import os
import runpy
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "tableS2"


def amax(x):
    return float(np.max(np.abs(np.asarray(x, dtype=float))))


def worst_ledger(*solutions):
    return max(max(s.ledger.check().values()) for s in solutions)


def convergence(n, error):
    """Errors on refined grids, and the observed order: the slope of log(error) against
    log(cells), overall and for each refinement."""
    n, e = np.asarray(n, float), np.abs(np.asarray(error, float))
    steps = np.log(e[:-1] / e[1:]) / np.log(n[1:] / n[:-1])
    order = float(-np.polyfit(np.log(n), np.log(e), 1)[0])
    return {"n": n.tolist(), "error": e.tolist(), "order": order, "order_steps": steps.tolist()}


# ---- one function per tutorial: its globals -> {"quantities": {...}, "convergence": {...}} ----


def t01(g):
    from scipy.special import gammainc

    ts, conc = g["ts"], np.asarray(g["sol_chain"]["conc"])
    erlang = [amax(g["arrival"](n) - 100 * gammainc(n, n * ts / 2.0)) for n in (1, 4, 16, 64)]
    inner = g["inner"]
    return {"quantities": {
        "chain_vs_axis": float(g["difference"]),
        "chain_vs_axis_rel": float(g["difference"]) / float(conc.max()),
        "erlang_vs_gamma_cdf": max(erlang),
        "savic_rate": amax((g["rate"] - g["savic"])[inner]) / float(g["savic"].max()),
    }}


def t02(g):
    gut, t = g["gut"], g["t"]
    stomach = []
    for p in g["cases"].values():
        s = gut.solve((0, 24), times=t, params=p, rtol=1e-8, atol=1e-10)
        eta, b = p.get("eta", 0.360674), p.get("beta_ge", 1.0)
        in_stomach = np.sum(np.asarray(s["G"]) * (8.0 / 160), axis=1)
        stomach.append(amax(in_stomach - 100 * np.exp(-((t / eta) ** b))))
    f_si = []
    for p in g["drugs"].values():
        small = gut.solve((0, 200), times=[0, 200], params={**p, "CL": 0.0, "ka_col": 0.0})
        ka, r = p.get("ka_si", 0.185), p.get("ka_ratio", 0.333333)
        closed = 1 - np.exp(-ka * (199 / 60) * (r - 1) / np.log(r))
        f_si.append(abs(float(np.asarray(small["Ac"])[-1]) / 100 - closed))
    repeated = g["repeated"]
    return {"quantities": {
        "notation_vs_api": float(g["diff"]),
        "notation_vs_api_rel": float(g["diff"]) / float(np.max(g["sol"]["conc"])),
        "weibull_emptying": max(stomach),
        "f_si_closed_form": max(f_si),
        "cat_closed_form": abs(g["f"] - g["yu_amidon"]),
        "regimen_vs_sum": amax(repeated - g["summed"]),
        "regimen_vs_sum_rel": amax(repeated - g["summed"]) / float(np.max(repeated)),
    }}


def t03(g):
    import tenqsp as tq

    circ0, mtt = g["P"]["circ0"], g["P"]["MTT"]
    erlang = []
    for n in (3, 30):
        c = tq.Model.from_text(g["COHORT"].format(n=n, sharp="")).solve(
            (0, 3000), times=[3000], rtol=1e-10, atol=1e-13)
        erlang.append(abs(float(np.asarray(c["W"])[-1]) - mtt))
    return {
        "quantities": {
            "chain_vs_axis": max(g["diff_circ"], g["diff_prol"], g["diff_transit"]),
            "chain_vs_axis_rel": max(g["diff_circ"], g["diff_prol"], g["diff_transit"]) / circ0,
            "baseline_rhs": float(g["residual"]),
            "baseline_drift": float(g["drift"]),
            "erlang_mean_time": max(erlang),
            "nadir_vs_hand": abs(g["nadirs"][(3, "erlang")][0] - g["by_hand"][4].min()),
        },
        "convergence": {
            "sharp_mean_time": convergence([100, 200, 400], np.asarray(g["sharp_errors"]) / mtt)
        },
    }


def t04(g):
    t, rise = g["t"], g["rise"]
    erlang = []
    for n in (4, 16, 160):
        s = g["lifespan"](n, "erlang").solve((0, 8), times=t, rtol=1e-10, atol=1e-12)
        erlang.append(amax(np.asarray(s["R"]) - g["erlang_step_response"](n, t)))
    r_exact = g["r_exact"]
    return {
        "quantities": {
            "sharp_away_from_corner": float(g["away"]),
            "sharp_plateau": amax(g["plateau"] - g["kin1"] * g["TR"]),
            "erlang_gamma_closed_form": max(erlang),
            "erlang_gamma_closed_form_rel": max(erlang) / rise,
            "drug_closed_form": float(g["err"].max()) / float(r_exact.max() - g["kin0"] * g["TR"]),
            "drug_concentration": amax(np.asarray(g["drug"]["conc"]) - g["conc"]) / g["c0"],
        },
        "convergence": {"sharp_corner": convergence([40, 80, 160], np.asarray(g["errors"]) / rise)},
    }


def t05(g):
    F, av, liver, steady = g["F_exact"], g["availability"], g["liver"], g["steady_availability"]
    rn = g["RN"]
    tanks = [abs(steady(liver(n, "erlang", dispersion=False), {"RN": rn}) - (1 + rn / n) ** -n)
             for n in (1, 3, 10)]
    mean, var, out = [], [], []
    for dn in (0.05, 0.17, 0.5):
        s = g["bolus"].solve((0, 40), times=np.append(g["theta"], 40), params={"DN": dn},
                             rtol=1e-10, atol=1e-13)
        w1, w2 = float(np.asarray(s["W1"])[-1]), float(np.asarray(s["W2"])[-1])
        kf = g["kf"]
        var_exact = 2 * dn - 2 * dn**2 * (1 - np.exp(-1 / dn))
        mean.append(abs(w1 - 1 / kf - 1))
        var.append(abs((w2 - w1**2) - 1 / kf**2 - var_exact) / var_exact)
        out.append(abs(float(np.asarray(s["Out"])[-1]) - 1))
    limits = {k: abs(m / lim - 1) for k, (m, lim) in g["limits"].items()}
    q = g["Q"]
    return {
        "quantities": {
            "parallel_tube": limits["parallel tube"],
            "well_stirred": limits["well stirred"],
            "tanks_in_series": max(tanks),
            "tanks_in_series_rel": max(t / (1 + rn / n) ** -n for t, n in zip(tanks, (1, 3, 10))),
            "F_vs_dispersion_number": max(abs(f - av(d, rn)) for d, f in
                                          zip(g["dn_points"], g["model_points"], strict=True)),
            "clearance_vs_closed_form": max(abs(c - q * (1 - av(0.17, x / q))) / q
                                            for x, c in
                                            zip(g["fu_clint_points"], g["cl_points"], strict=True)),
            "bolus_mean_transit": max(mean),
            "bolus_variance": max(var),
            "bolus_recovered": max(out),
            "bolus_out_vs_F": abs(g["came_out"] - F),
            "ledger_out_plus_eliminated": abs(g["came_out"] + g["eliminated"] - 1),
        },
        "convergence": {
            "sharp_F": convergence([20, 40, 80, 160], g["errors"]["sharp"]),
            "erlang_F": convergence([20, 40, 80, 160], g["errors"]["erlang"]),
        },
    }


def t06(g):
    pbpk, hep = g["pbpk"], g["hepatic_clearance"]
    ps_err = [abs(g["simulated"] / g["formula"] - 1)]  # PS = 1000, the loop's last value
    for ps in (10.0, 100.0):
        s = pbpk.solve((0, 800), times=[800], regimen=g["infusion"],
                       params={"PS1": ps, "PS2": ps}, **g["TIGHT"])
        ps_err.append(abs(10 / float(s["Cart"][-1]) / hep(PS1=ps, PS2=ps) - 1))
    keep = [o for o, name in enumerate(g["ORGANS"]) if name != "liver"]
    C_end, kp = g["C_end"], g["KP"]
    return {"quantities": {
        "field_vs_odes": float(g["diff"]),
        "mass_balance": amax(g["total"] - 100),
        "mass_balance_rel": amax(g["total"] - 100) / 100,
        "ledger": max(g["sol"].ledger.check().values()),
        "auc": abs(g["auc"] / (100 / g["CL_H"]) - 1),
        "css": abs(g["c_art"] / (10 / g["CL_H"]) - 1),
        "kp_tissue": max(abs(g["ratio"][-1, o] / g["kp_tissue"][o] - 1) for o in keep),
        "cells_vs_kp": max(abs(C_end[o, 2] / C_end[o, 0] / kp[o] - 1) for o in keep),
        "clearance_vs_ps": max(ps_err),
    }}


def t07(g):
    sol, ref = g["sol"], g["ref"]
    standard = []
    for j, name in enumerate(["T", "I", "V"]):
        got, want = np.asarray(sol[name]), ref.y[j]
        standard.append(amax((got - want) / (np.abs(want) + 1e-12 * np.abs(want).max())))
    r0 = []
    for eps_rt, eps_pi in ((0.0, 0.0), (0.3, 0.2)):
        exact = (1 - eps_rt) * (1 - eps_pi) * g["K"] * g["T0"] / g["C"] * g["burst"]
        grid = g["r0_model"].solve((0, 1e-6), times=[1e-6],
                                   params={"eps_rt": eps_rt, "eps_pi": eps_pi}, ledger=False)
        r0.append(abs(float(grid["R0"][-1]) / exact - 1))
    v_star, b_star = g["V_star"], g["b_star"]
    return {"quantities": {
        "standard_model": max(standard),
        "r0_quadrature": max(r0),
        "set_point": abs(g["T_star"] / (g["T0"] / g["R0"]) - 1),
        "age_profile":
            amax(g["set_point"].state["i"] - b_star * g["survival"](g["a_mid"])) / b_star,
        "growth_rate": abs(g["slope"] / g["r"] - 1),
        "perfect_pi": amax(g["viral"]["perfect PI"] - g["pi_exact"]) / v_star,
        "perfect_rti": amax(g["viral"]["perfect RTI"] - g["rti_exact"]) / v_star,
    }}


def t08(g):
    return {
        "quantities": {
            "no_births_decay": float(g["gap"]),
            "euler_lotka": abs(g["r_sim"] / g["r"] - 1),
            "stable_age_distribution": float(g["l1"]),
            "rbc_lifespan": amax(np.asarray(g["rbc"]["total"]).ravel() / g["expected"] - 1),
        },
        "convergence": {"characteristics": convergence([120, 240, 480], g["errors"])},
    }


def t09(g):
    nielsen = []
    for c, s in g["kill_curves"].items():
        ref = g["nielsen_by_hand"](c)
        nielsen.append(amax((np.asarray(s["B"]) - ref) / np.maximum(ref, 1.0)))
    lam = g["lam"]
    return {"quantities": {
        "matrix_exponential": float(g["rel"]),
        "fast_slope": abs(g["fast"] / lam[0] - 1),
        "slow_slope": abs(g["slow"] / lam[1] - 1),
        "nielsen_vs_hand": max(nielsen),
        "fokker_planck_mass": amax(g["totals"] / 1e6 - 1),
        "fokker_planck_steady": float(g["err"]),
        "two_wells_vs_two_states": amax(g["awake"] - g["two_state"]),
        "ledger": max(g["sol_k"].ledger.check().values()),
        "late_slope": abs(g["slope"] / g["expected"] - 1),
    }}


def t10(g):
    slab, z_faces, L = g["slab"], g["z_faces"], g["L"]
    profile_err = []
    for phi in (0.5, 2.0, 5.0):
        kr = phi**2 * g["DE"] / L**2
        sol = slab.solve((0, 60), times=[60], params={"kr": kr}, **g["TIGHT"])
        profile = np.asarray(sol["C"])[-1]
        exact = np.diff(np.sinh(phi * z_faces / L)) / (phi / L) / np.diff(z_faces) / np.cosh(phi)
        profile_err.append(amax(profile - exact))
    into = g["into_biofilm"]
    return {"quantities": {
        "cosh_profile": max(profile_err),
        "effectiveness_factor": max(abs(e / (np.tanh(p) / p) - 1) for p, e in g["etas"].items()),
        "crank_series": amax(g["mean_in"] - g["crank"]),
        "ledger_surface": amax(into + g["out_of_liquid"]) / amax(into),
        "ledger_books": amax(g["flowed_in"] - g["accounted"]) / float(g["flowed_in"][-1]),
        "steady_liquid": abs(g["cb"] / (10 * 50 / (50 + g["G"])) - 1),
    }}


def t11(g):
    pushed = {}
    for delta, (s, _) in g["pushed"].items():
        c, _ = g["front_speed"](s, 4.0)
        cs, _ = g["minimal_speed"](dict(g["V"], delta=delta))
        pushed[delta] = c / cs - 1
    return {"quantities": {
        "growth_rate": abs(g["rate"] / g["r"] - 1),
        "fisher_speed": abs(g["c_fisher"] / g["c_star_f"] - 1),
        "shallow_front_speed": abs(g["c_shallow"] / g["speed"](g["V"], 0.1) - 1),
        "pushed_front_slow_decay": pushed[0.05],
        "pushed_front_fast_decay": pushed[3.0],
        "ledger": worst_ledger(g["sol"], g["sol_s"], *(s for s, _ in g["pushed"].values())),
    }}


def t12(g):
    balance, c = g["balance"], g["c"]
    size_err, mean_err = [], []
    for vbar, s in g["runs"].items():
        exp_size, exp_mean, _ = balance(c, vbar)
        size_err.append(abs(float(np.asarray(s["size"])[-1]) / exp_size - 1))
        mean_err.append(abs(float(np.asarray(s["mean_x"])[-1]) - exp_mean))
    m5 = g["m5"]
    return {
        "quantities": {
            "notation_vs_api_identical":
                bool(g["tumour"].to_ir() == g["drug_tolerance_model"]().to_ir()),
            "theorem1_size": abs(g["size0"] / g["rho_eq"] - 1),
            "theorem1_mean": abs(g["mean0"]),
            "theorem1_variance": abs(g["var0"] / g["var_eq"] - 1),
            "theorem2_size": max(size_err),
            "theorem2_mean": max(mean_err),
            "three_odes_size": amax(g["size_pde"] / g["rho3"] - 1),
            "three_odes_mean": amax(g["mean_pde"] - g["m3"]),
            "doses_mean_160": amax(g["mean5"] - m5),
            "doses_mean_320": amax(np.asarray(g["fine"]["mean_x"]) - m5),
            "ledger": worst_ledger(g["sim"].solution, g["b"]),
            "bounded_grid_change": float(g["change"]),
        },
        "convergence": {"doses_size": convergence([160, 320], [g["err_coarse"], g["err_fine"]])},
    }


EXTRACT = {f"{k:02d}": f for k, f in
           enumerate([t01, t02, t03, t04, t05, t06, t07, t08, t09, t10, t11, t12], 1)}


def run_one(path):
    """Run one tutorial from its folder and write its numbers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tenqsp

    path = Path(path)
    key = path.name[:2]
    os.chdir(path.parent)
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):  # the tutorial's own printout
        g = runpy.run_path(str(path), run_name="__main__")
    seconds = time.perf_counter() - start
    plt.close("all")
    out = EXTRACT[key](g)
    out.update(tutorial=path.name, tenqsp=tenqsp.__version__, seconds=round(seconds, 1),
               seconds_extraction=round(time.perf_counter() - start - seconds, 1))
    (RESULTS / f"{key}.json").write_text(json.dumps(out, indent=1) + "\n")
    return key, seconds


def sdist_tutorials():
    """docs/tutorials from the source distribution of the installed TenQSP, from PyPI."""
    import tenqsp

    version = tenqsp.__version__
    cache = Path(tempfile.gettempdir()) / f"tenqsp-{version}-sdist"
    folder = cache / f"tenqsp-{version}" / "docs" / "tutorials"
    if not folder.is_dir():
        cache.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "pip", "download", f"tenqsp=={version}",
                        "--no-binary", ":all:", "--no-deps", "-d", str(cache)], check=True)
        with tarfile.open(cache / f"tenqsp-{version}.tar.gz") as tar:
            tar.extractall(cache, filter="data")
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tutorials", type=Path,
                        help="folder with the tutorials (default: from PyPI)")
    parser.add_argument("--only", nargs="*", help="tutorial numbers, e.g. 01 05")
    args = parser.parse_args()
    folder = (args.tutorials or sdist_tutorials()).resolve()
    paths = sorted(folder.glob("[01][0-9]_*.py"))
    if args.only:
        paths = [p for p in paths if p.name[:2] in args.only]
    RESULTS.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    # tutorial 02 is the longest: start the longest first
    paths.sort(key=lambda p: p.name[:2] not in ("02", "11", "12"))
    with ProcessPoolExecutor(max_workers=len(paths), mp_context=get_context("spawn")) as pool:
        jobs = [pool.submit(run_one, str(p)) for p in paths]
        for job in as_completed(jobs):
            key, seconds = job.result()
            print(f"tutorial {key}: {seconds:.0f} s", flush=True)
    print(f"all done in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
