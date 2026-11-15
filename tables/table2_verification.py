# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Table 2: the checks of TenQSP's 12 reference models (its tutorials), from
results/table2/*.json; writes out/table2.md and out/table2.csv (seconds).

    python tables/table2_runs.py           # the computations, about 5 min
    python tables/table2_verification.py

Each row is one check a tutorial makes against a known result. "Measured" is the number the
tutorial's check measures, in the same measure (absolute or relative, as stated); "Bound" is
the tolerance the tutorial asserts. Where a check is a grid-convergence study, the error is
on the finest grid and the observed order is the slope of log(error) against log(cells).
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "table2"
OUT = ROOT / "out"

MODELS = {
    "01": "Transit compartments (Savic et al. 2007)",
    "02": "Gut absorption (Willmann et al. 2003; CAT model, Yu & Amidon 1999)",
    "03": "Myelosuppression (Friberg et al. 2002)",
    "04": "Lifespan indirect response (Krzyzanski et al. 1999)",
    "05": "Hepatic dispersion (Roberts & Rowland 1986)",
    "06": "Whole-body PBPK (Jones & Rowland-Yeo 2013)",
    "07": "HIV infection age (Nelson et al. 2004; Rong et al. 2007)",
    "08": "Age-structured population (McKendrick 1926)",
    "09": "Persisters and dormancy, Fokker-Planck (Balaban et al. 2004; Nielsen et al. 2007)",
    "10": "Biofilm penetration (Stewart 1996)",
    "11": "Phage plaque fronts (Yin & McCaskill 1992; Fort & Mendez 2002)",
    "12": "Tumour drug-tolerance continuum (Lorenzi et al. 2016; Chisholm et al. 2015)",
}

# (tutorial, key, what is checked, measure, the tutorial's bound); a key in "convergence" is a
# grid-convergence study, any other key is a single number
ROWS = [
    ("01", "chain_vs_axis", "chain of 4 compartments (ODEs) = erlang axis on 4 cells",
     "abs., mg/L", "1e-9"),
    ("01", "erlang_vs_gamma_cdf", ("N erlang cells = Erlang(N) delay: arrival vs gamma CDF "
     "(N = 1, 4, 16, 64)"), "abs., mg of 100 mg", "1e-4"),
    ("01", "savic_rate", "arrival rate vs Savic's closed-form input",
     "rel. to peak rate (numerical derivative)", "0.01"),
    ("02", "notation_vs_api", "notation = Python API, same model", "abs., mg/L", "1e-7"),
    ("02", "weibull_emptying", "drug in stomach = Weibull survival function (2 cases)",
     "abs., mg of 100 mg", "1"),
    ("02", "f_si_closed_form", ("fraction absorbed in small intestine = closed form "
     "(log-linear ka, plug flow; 2 drugs)"), "abs., fraction of dose", "3e-3"),
    ("02", "cat_closed_form", "7 erlang cells = CAT model, F = 1 - (1 + ka T/7)^-7",
     "abs., fraction of dose", "1e-6"),
    ("02", "regimen_vs_sum", "14-dose regimen = sum of shifted single doses", "abs., mg/L",
     "1e-6"),
    ("03", "chain_vs_axis", "3 erlang cells = published model (ODEs, scipy)",
     "abs., 10^9 cells/L", "5e-9"),
    ("03", "baseline_rhs", "no drug: right-hand side at baseline = 0", "abs.", "1e-12"),
    ("03", "baseline_drift", "no drug: Circ stays at Circ0 for 2000 h", "abs., 10^9 cells/L",
     "1e-9"),
    ("03", "erlang_mean_time", "mean maturation time = MTT (3 and 30 erlang cells)", "abs., h",
     "1e-6"),
    ("03", "sharp_mean_time", "sharp scheme: mean maturation time -> MTT (100-400 cells)",
     "rel. to MTT", "1e-3 on 100 cells"),
    ("04", "sharp_corner", ("sharp: step response vs closed form, largest error "
     "(at the corner t = T_R; 40-160 cells)"), "rel. to the rise", "0.01 on 160 cells"),
    ("04", "sharp_away_from_corner", "sharp, 160 cells: step response away from the corner",
     "abs., cells", "1e-4"),
    ("04", "sharp_plateau", "sharp, 160 cells: plateau = kin1 T_R", "abs., cells", "1e-3"),
    ("04", "erlang_gamma_closed_form", ("N erlang cells = gamma lifespans: closed-form step "
     "response (N = 4, 16, 160)"), "abs., cells", "1e-6"),
    ("04", "drug_closed_form", ("drug-stimulated production: R = integral of k_in over the "
     "last T_R (400 cells)"), "rel. to the peak rise", "0.01"),
    ("05", "sharp_F", "availability F vs closed form (Roberts-Rowland), sharp, 20-160 cells",
     "abs., F", "1e-6 on 160 cells"),
    ("05", "erlang_F", "availability F vs closed form, erlang, 20-160 cells", "abs., F",
     "first order"),
    ("05", "parallel_tube", "D_N = 0.001: F = exp(-R_N) (parallel tube)", "rel.", "0.01"),
    ("05", "well_stirred", "D_N = 1000: F = 1/(1 + R_N) (well stirred)", "rel.", "0.01"),
    ("05", "tanks_in_series", ("N erlang cells, no dispersion = tanks in series "
     "(N = 1, 3, 10)"), "abs., F", "1e-8"),
    ("05", "F_vs_dispersion_number", "F at 5 values of D_N vs closed form", "abs., F", "1e-4"),
    ("05", "bolus_mean_transit", "bolus: mean transit time = 1 (D_N = 0.05, 0.17, 0.5)",
     "abs., V/Q", "1e-4"),
    ("05", "bolus_variance", "bolus: variance = 2 D_N - 2 D_N^2 (1 - exp(-1/D_N))", "rel.",
     "1e-3"),
    ("05", "bolus_out_vs_F", "bolus with elimination: amount out = F x dose",
     "abs., fraction of dose", "1e-4"),
    ("05", "ledger_out_plus_eliminated", "ledger: out + eliminated = dose",
     "abs., fraction of dose", "1e-9"),
    ("06", "field_vs_odes", "field over 9 organs x 3 spaces = 29 hand-written ODEs", "rel.",
     "1e-6"),
    ("06", "mass_balance", "tissues + blood + eliminated (ledger) = dose", "abs., mg of 100 mg",
     "1e-6"),
    ("06", "ledger", "ledger: largest unexplained part", "rel.", "1e-10"),
    ("06", "auc", "AUC = Dose/CL_H (membranes in series)", "rel.", "1e-8"),
    ("06", "css", "infusion: Css = R/CL_H", "rel.", "1e-8"),
    ("06", "kp_tissue", "tissue:plasma = Kp_tissue (8 non-eliminating organs)", "rel.", "1e-7"),
    ("06", "clearance_vs_ps", "CL_H at PS = 10, 100, 1000 L/h = series formula", "rel.",
     "1e-7"),
    ("07", "standard_model", "constant rates = standard 3-ODE model (T, I, V)", "rel.", "1e-6"),
    ("07", "r0_quadrature", "R0 on the grid vs quadrature (2 drug settings)", "rel.", "2e-3"),
    ("07", "set_point", "set point T* = T0/R0", "rel.", "2e-3"),
    ("07", "age_profile", "set-point age profile = b* S(a)", "rel. to b*", "1e-3"),
    ("07", "growth_rate", "early growth rate = root of the renewal equation", "rel.", "3e-3"),
    ("07", "perfect_pi", "perfect PI: V = V* exp(-c t)", "rel. to V*", "1e-6"),
    ("07", "perfect_rti", "perfect RTI: V from the characteristics", "rel. to V*", "1.5e-2"),
    ("08", "no_births_decay", "no births: N = N0 exp(-mu t)", "rel.", "1e-9"),
    ("08", "characteristics", ("age-dependent death: density vs characteristics "
     "(120-480 cells)"), "L1, rel.", "1e-2, ratios > 3"),
    ("08", "euler_lotka", "growth rate = Euler-Lotka root", "rel.", "1e-5"),
    ("08", "stable_age_distribution", "age profile = stable age distribution", "L1", "1e-4"),
    ("08", "rbc_lifespan", "fixed lifespan: total = kin min(t, T_R)", "rel.", "1e-8"),
    ("09", "matrix_exponential", "two-state switch (Balaban) = matrix exponential", "rel.",
     "1e-7"),
    ("09", "slow_slope", "late slope of log(total) = slow eigenvalue", "rel.", "1e-6"),
    ("09", "nielsen_vs_hand", ("Nielsen model (switching depends on the total) = "
     "hand-written ODEs (5 concentrations)"), "rel.", "1e-6"),
    ("09", "fokker_planck_mass", "Fokker-Planck, no growth or death: total conserved", "rel.",
     "1e-12"),
    ("09", "fokker_planck_steady", "steady state = closed form (80 cells)", "rel. to peak",
     "5e-3"),
    ("09", "two_wells_vs_two_states", ("two wells: awake fraction = two-state model with rates "
     "from the landscape"), "abs., fraction", "1e-3"),
    ("09", "late_slope", ("growth and kill: late slope = leading eigenvalue (finite "
     "differences, 1000 cells)"), "rel.", "0.01"),
    ("09", "ledger", "ledger, growth and kill: largest unexplained part", "rel.", "1e-10"),
    ("10", "cosh_profile", "steady profile = cosh closed form (phi = 0.5, 2, 5)",
     "abs., fraction of Cb", "3e-3"),
    ("10", "effectiveness_factor", "effectiveness factor = tanh(phi)/phi", "rel.", "3e-3"),
    ("10", "crank_series", "no reaction: mean concentration = Crank's series",
     "abs., fraction of Cb", "2e-3"),
    ("10", "steady_liquid", "biofilm as a sink: liquid = Cin CL/(CL + G)", "rel.", "1e-3"),
    ("10", "ledger_surface", "ledger: entered biofilm = left liquid", "rel.", "1e-12"),
    ("10", "ledger_books", "ledger: inflow = liquid + biofilm + destroyed + washed out", "rel.",
     "1e-9"),
    ("11", "growth_rate", "phage growth rate = root of the characteristic equation", "rel.",
     "3e-3"),
    ("11", "fisher_speed", "no latent period: front speed = 2 sqrt(D r) (Fisher-KPP)", "rel.",
     "0.03"),
    ("11", "shallow_front_speed", "shallow start: front speed = c(lambda), linear theory",
     "rel.", "0.01"),
    ("11", "pushed_front_slow_decay", ("steep start, delta = 0.05/h: front speed vs minimal "
     "speed c*"), "(c - c*)/c*", "-3 % to +20 %"),
    ("11", "pushed_front_fast_decay", "steep start, delta = 3/h: front speed vs c*",
     "(c - c*)/c*", "-3 % to +20 %"),
    ("11", "ledger", "ledger (4 runs): largest unexplained part", "rel.", "1e-9"),
    ("12", "notation_vs_api_identical", "notation = Python API, same model",
     "model descriptions compared", "identical"),
    ("12", "theorem1_size", "no drug: size = Theorem 1 balance", "rel.", "2e-4"),
    ("12", "theorem1_mean", "no drug: mean level = 0", "abs.", "1e-4"),
    ("12", "theorem1_variance", "no drug: variance = sqrt(beta/eta)", "rel.", "2e-3"),
    ("12", "theorem2_size", "constant drug: size = Theorem 2 balance (with and without drift)",
     "rel.", "1e-3"),
    ("12", "theorem2_mean", "constant drug: mean level = balance", "abs.", "1e-3"),
    ("12", "three_odes_size", "constant drug: PDE = three ODEs, size (160 cells)", "rel.",
     "2e-3"),
    ("12", "three_odes_mean", "constant drug: PDE = three ODEs, mean level (160 cells)",
     "abs.", "1e-3"),
    ("12", "doses_size", "doses and holiday: PDE vs three ODEs, size (160, 320 cells)", "rel.",
     "1.5e-3 on 320 cells"),
    ("12", "ledger", "ledger (2 runs): largest unexplained part", "rel.", "1e-10"),
    ("12", "bounded_grid_change", "bounded level: size on 100 vs 200 cells", "rel.", "1e-3"),
]


def load():
    return {p.stem: json.loads(p.read_text()) for p in sorted(RESULTS.glob("[01][0-9].json"))}


def measured(entry):
    """The number shown, and the observed order (for a convergence study)."""
    if isinstance(entry, dict):
        steps = ", ".join(f"{s:.2f}" for s in entry["order_steps"])
        order = f"{entry['order']:.2f}" + (f" ({steps})" if len(entry["order_steps"]) > 1 else "")
        return entry["error"][-1], order
    return entry, ""


def fmt(value, key):
    if isinstance(value, bool):
        return "identical" if value else "different"
    if key.startswith("pushed"):
        return f"{value:+.1%}"
    return "0" if value == 0 else f"{value:.1e}"


def main():
    data = load()
    OUT.mkdir(exist_ok=True)
    head = ["Model (reference)", "What is checked", "Measured", "Measure", "Observed order",
            "Bound in tutorial"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    rows, last = [], None
    for tut, key, what, measure, bound in ROWS:
        if tut not in data:
            continue
        d = data[tut]
        value, order = measured({**d["quantities"], **d.get("convergence", {})}[key])
        rows.append([MODELS[tut], what, value, measure, order, bound])
        model = f"**{tut}** {MODELS[tut]}" if tut != last else ""
        last = tut
        lines.append(f"| {model} | {what} | {fmt(value, key)} | {measure} | {order} | {bound} |")
    versions = sorted({d["tenqsp"] for d in data.values()})
    seconds = sum(d["seconds"] for d in data.values())
    lines += ["", (f"TenQSP {', '.join(versions)}; {len(data)} tutorials, {seconds:.0f} s of "
              "run time in all (one process each). Observed order: slope of log(error) against "
              "log(cells) over all grids, and for each doubling in brackets.")]
    (OUT / "table2.md").write_text("\n".join(lines) + "\n")
    with open(OUT / "table2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
