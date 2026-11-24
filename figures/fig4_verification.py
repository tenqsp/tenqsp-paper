# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Figure 4: verification on TenQSP's 12 reference models (its tutorials), drawn from
results/tableS2/*.json into out/fig4_verification.png and .pdf (seconds).

    python tables/tableS2_runs.py           # the computations, about 5 min
    python figures/fig4_verification.py

(A) error against the number of cells for the tutorials with a grid-convergence study,
with reference slopes 1 and 2; (B) the largest unexplained part of each conservation ledger
and mass balance; (C) the differences between models that must be identical (an axis and the
chain of compartments it replaces, a field and the hand-written ODEs, ...). Drawn at print
size (printstyle.py).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import printstyle as ps

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "tableS2"
OUT = ROOT / "out"
EPS = np.finfo(float).eps
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
BAR = "#2a78d6"

# (tutorial, key, label)
CONVERGENCE = [
    ("03", "sharp_mean_time", "03 Friberg: mean maturation time, sharp"),
    ("04", "sharp_corner", "04 lifespan: step response at its corner, sharp"),
    ("05", "sharp_F", "05 dispersion: availability F, sharp"),
    ("05", "erlang_F", "05 dispersion: availability F, erlang"),
    ("08", "characteristics", "08 McKendrick: density vs characteristics, sharp"),
    ("12", "doses_size", "12 tolerance: size vs three ODEs, sharp"),
]
LEDGERS = [
    ("05", "ledger_out_plus_eliminated", "05 dispersion: out + eliminated = dose"),
    ("06", "ledger", "06 PBPK: ledger"),
    ("06", "mass_balance_rel", "06 PBPK: tissues + blood + eliminated = dose"),
    ("09", "fokker_planck_mass", "09 Fokker-Planck: total conserved"),
    ("09", "ledger", "09 dormancy, growth and kill: ledger"),
    ("10", "ledger_surface", "10 biofilm: entered biofilm = left liquid"),
    ("10", "ledger_books", "10 biofilm: inflow = all accounted for"),
    ("11", "ledger", "11 plaque: ledger (4 runs)"),
    ("12", "ledger", "12 tolerance: ledger (2 runs)"),
]
EQUIVALENCES = [
    ("01", "chain_vs_axis_rel", "01 transit chain = axis, 4 cells"),
    ("02", "notation_vs_api_rel", "02 gut: notation = Python API"),
    ("02", "cat_closed_form", "02 gut: 7 erlang cells = CAT model"),
    ("02", "regimen_vs_sum_rel", "02 gut: regimen = sum of single doses"),
    ("03", "chain_vs_axis_rel", "03 Friberg ODEs = axis, 3 cells"),
    ("04", "erlang_gamma_closed_form_rel", "04 erlang cells = gamma lifespans"),
    ("05", "tanks_in_series_rel", "05 erlang cells = tanks in series"),
    ("06", "field_vs_odes", "06 PBPK field = 29 hand-written ODEs"),
    ("07", "standard_model", "07 HIV, constant rates = 3 ODEs"),
    ("09", "matrix_exponential", "09 two-state switch = matrix exponential"),
    ("09", "nielsen_vs_hand", "09 Nielsen model = hand-written ODEs"),
    ("12", "notation_vs_api_identical", "12 tolerance: notation = Python API"),
]


def load():
    return {p.stem: json.loads(p.read_text()) for p in sorted(RESULTS.glob("[01][0-9].json"))}


def convergence_panel(ax, data):
    for k, (tut, key, label) in enumerate(CONVERGENCE):
        c = data[tut]["convergence"][key]
        ax.loglog(c["n"], c["error"], marker=MARKERS[k], ms=3, lw=1, color=COLORS[k],
                  label=f"{label} (order {c['order']:.1f})")
    n = np.array([250.0, 900.0])  # reference slopes, in an empty corner
    for slope, anchor, text in ((1, 3e-5, "slope 1"), (2, 6e-6, "slope 2")):
        y = anchor * (n / n[0]) ** -slope
        ax.loglog(n, y, color="0.55", ls="--", lw=0.7)
        ax.annotate(text, (n[1], y[1]), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=ps.SIZE - 1, color="0.35")
    ax.set_xlim(12, 1500)
    ax.set_xlabel("Number of cells along the axis")
    ax.set_ylabel("Error on the grid (each check's own measure)")
    ax.figure.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=ps.SIZE - 0.5,
                     bbox_to_anchor=(0.5, 0.005), columnspacing=1.5)


def bar_panel(ax, data, rows, xlabel, floor):
    labels, values, notes = [], [], []
    for tut, key, label in rows:
        v = data[tut]["quantities"][key]
        labels.append(label)
        if isinstance(v, bool):
            values.append(np.nan)
            notes.append("identical model description" if v else "different")
        else:
            values.append(max(v, floor))
            notes.append("0 (exact)" if v == 0 else "")
    y = np.arange(len(rows))[::-1]
    ax.barh(y, np.nan_to_num(values, nan=floor) - floor, left=floor, height=0.6, color=BAR)
    for yy, v, note in zip(y, values, notes, strict=True):
        text = note or f"{v:.0e}"
        ax.annotate(text, (np.nan_to_num(v, nan=floor), yy), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=ps.SIZE - 1, color="0.25")
    ax.axvline(EPS, color="0.55", ls=":", lw=0.7)
    ax.annotate("machine epsilon", (EPS, y[0] + 0.7), fontsize=ps.SIZE - 1, color="0.35", ha="center")
    ax.set_xscale("log")
    ax.set_xlim(floor, 1e-5)
    ax.set_ylim(-0.7, len(rows) - 0.1)
    ax.set_yticks(y, labels, fontsize=ps.SIZE - 0.5)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", length=0)


def main():
    data = load()
    OUT.mkdir(exist_ok=True)
    ps.setup()
    fig = plt.figure(figsize=(ps.WIDTH, 4.6))
    grid = fig.add_gridspec(2, 2, width_ratios=[0.8, 1], height_ratios=[1, 1.3], hspace=0.6, wspace=1.25,
                            left=0.085, right=0.97, top=0.95, bottom=0.25)
    a = fig.add_subplot(grid[:, 0])
    b = fig.add_subplot(grid[0, 1])
    c = fig.add_subplot(grid[1, 1])
    convergence_panel(a, data)
    bar_panel(b, data, LEDGERS, "Largest unexplained part, relative", 1e-17)
    bar_panel(c, data, EQUIVALENCES, "Largest difference, relative", 1e-17)
    for ax in (a, b, c):
        ax.spines[["top", "right"]].set_visible(False)
    a.set_title("A  Grid convergence")
    b.set_title("B  Conservation ledgers")
    c.set_title("C  Models that must be identical")
    ps.check(fig, [a, b, c] + fig.legends)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig4_verification.{ext}")
    print(f"wrote {OUT / 'fig4_verification.png'}")


if __name__ == "__main__":
    main()
