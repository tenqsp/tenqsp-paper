# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Figure 5: a whole-body PBPK model as one field over organs and spaces. Drawn from
results/fig5 (run fig5_runs.py first), at the journal's full width (7 in), text at 6-7 pt; it
stops if any text leaves its panel.

    python figures/fig5_pbpk.py      # writes out/fig5_pbpk.png and .pdf

A  the model's structure, from models/pbpk_wholebody.tq: the field C(organ, space) as nine
   organs of three spaces each, the blood pools, the blood flows and the portal drainage
B  the field C(organ, space) after a 100 mg IV dose, at three times (mg/L)
C  arterial blood and whole-organ concentrations; broken lines: the liver's three spaces
D  hepatic clearance against the membranes' permeability: simulated (dots), the series
   formula (line) and its limit, the well-stirred liver (dashed)
E  the exact checks: relative error of each
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pbpk as pb
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

DPI = 300
SIZE = 8  # points, at print size
SPACE_COLORS = {"v": "#f4c7c3", "i": "#fde9b8", "c": "#cfe3f5"}
VEN, ART = "#7aa6d6", "#e07a6f"  # venous and arterial blood


def arrow(ax, start, end, color="0.25", lw=0.7, **kw):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=5, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, **kw))


def structure(ax):
    """Panel A: venous blood on the left, arterial on the right, one row per organ. Each organ
    is a vascular strip along its top (where blood enters and leaves) over its interstitial and
    cellular spaces."""
    ax.set_xlim(0, 10)
    ax.set_ylim(-2.6, 11.0)
    ax.axis("off")
    x0, x1 = 2.75, 7.25  # organ boxes
    xi, xc = 5.0, 6.125  # left edges of the interstitial and cellular cells
    h, gap, top = 0.84, 0.24, 10.0
    rows = {}
    for k, organ in enumerate(pb.ORGANS):
        y = top - k * (h + gap) - h  # bottom of the row
        rows[organ] = y
        hv = 0.28 * h
        ax.add_patch(Rectangle((x0, y + h - hv), x1 - x0, hv, fc=SPACE_COLORS["v"], ec="0.3", lw=0.5))
        ax.add_patch(Rectangle((x0, y), xi - x0, h - hv, fc="white", ec="0.3", lw=0.5))
        ax.add_patch(Rectangle((xi, y), xc - xi, h - hv, fc=SPACE_COLORS["i"], ec="0.3", lw=0.5))
        ax.add_patch(Rectangle((xc, y), x1 - xc, h - hv, fc=SPACE_COLORS["c"], ec="0.3", lw=0.5))
        ax.text((x0 + xi) / 2, y + (h - hv) / 2, organ, ha="center", va="center", fontsize=SIZE)
        if k == 0:  # labels of the spaces, once
            ax.text(x0 + 0.1, y + h - hv / 2, "v", ha="left", va="center", fontsize=SIZE, style="italic")
            for x, lab in (((xi + xc) / 2, "i"), ((xc + x1) / 2, "c")):
                ax.text(x, y + (h - hv) / 2, lab, ha="center", va="center", fontsize=SIZE, style="italic")
    yv = {o: rows[o] + h - 0.14 * h for o in pb.ORGANS}  # middle of each vascular strip
    y_low, y_high = rows["liver"] - 0.05, top + 0.05
    for x, c, name in ((0.35, VEN, "venous\nblood"), (8.65, ART, "arterial\nblood")):
        ax.add_patch(FancyBboxPatch((x, y_low), 1.0, y_high - y_low, boxstyle="round,pad=0,rounding_size=0.3",
                                    fc=c, ec="none", alpha=0.85))
        ax.text(x + 0.5, (y_low + y_high) / 2, name, ha="center", va="center", fontsize=SIZE, rotation=90,
                color="white", weight="bold")
    # lung: venous -> lung -> arterial (the whole cardiac output)
    arrow(ax, (1.35, yv["lung"]), (x0, yv["lung"]), color=VEN, lw=1.0)
    arrow(ax, (x1, yv["lung"]), (8.65, yv["lung"]), color=ART, lw=1.0)
    ax.text((x1 + 8.65) / 2, yv["lung"] + 0.12, "Q$_{CO}$", ha="center", va="bottom", fontsize=SIZE)
    for organ in pb.ORGANS[1:]:
        arrow(ax, (8.65, yv[organ]), (x1, yv[organ]), color=ART)
        if organ not in ("gut", "spleen"):
            arrow(ax, (x0, yv[organ]), (1.35, yv[organ]), color=VEN)
    # portal drainage: gut and spleen into the liver's vascular space
    xp = 2.2
    for organ in ("gut", "spleen"):
        ax.plot([x0, xp], [yv[organ]] * 2, c="0.35", lw=0.7)
    y_in = rows["liver"] + h + 0.11
    ax.plot([xp, xp, 3.2], [yv["gut"], y_in, y_in], c="0.35", lw=0.7)
    arrow(ax, (3.2, y_in), (3.2, rows["liver"] + h), color="0.35")
    ax.text(xp - 0.08, (yv["gut"] + yv["spleen"]) / 2, "portal", ha="right", va="center", fontsize=SIZE,
            rotation=90)
    # metabolism in the liver cells, and the dose
    xm = (xc + x1) / 2
    arrow(ax, (xm, rows["liver"]), (xm, rows["liver"] - 0.55), color="0.25")
    ax.text(xm + 0.12, rows["liver"] - 0.3, "f$_u$CL$_{int}$", ha="left", va="center", fontsize=SIZE)
    arrow(ax, (0.85, y_high + 0.35), (0.85, y_high), color="0.25")
    ax.text(0.85, y_high + 0.38, "IV dose", ha="center", va="bottom", fontsize=SIZE)
    # what the field is
    ax.text(5.0, -1.5, "field C(organ, space): 9 × 3 = 27 states\n"
            "v↔i at PS$_1$, i↔c at PS$_2$ (cells at K$_p$ × interstitium)\n"
            "blood pools: 2 compartments",
            ha="center", va="center", fontsize=SIZE, linespacing=1.3)


def check_inside(fig, ax, texts, where):
    fig.canvas.draw()
    box = ax.get_window_extent() if where == "axes" else fig.bbox
    for t in texts:
        bb = t.get_window_extent()
        if bb.x0 < box.x0 - 1 or bb.x1 > box.x1 + 1 or bb.y0 < box.y0 - 1 or bb.y1 > box.y1 + 1:
            raise SystemExit(f"text leaves its {where} in {ax.get_title(loc='left')!r}: {t.get_text()[:40]!r}")


def main():
    iv = np.load(pb.RESULTS / "iv.npz")
    cl = np.load(pb.RESULTS / "clearance.npz")
    checks = json.loads((pb.RESULTS / "checks.json").read_text())
    plt.rcParams.update({"font.size": SIZE, "axes.titlesize": SIZE, "axes.titleweight": "bold",
                         "axes.labelsize": SIZE, "legend.fontsize": SIZE, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0})
    # at 8 pt: A and B on top, C and D below them, E on its own row
    fig = plt.figure(figsize=(7.0, 8.6), dpi=DPI)
    top = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.3, left=0.01, right=0.9, top=0.96,
                           bottom=0.58)
    mid = fig.add_gridspec(1, 2, wspace=0.3, left=0.1, right=0.985, top=0.5, bottom=0.27)
    low = fig.add_gridspec(1, 1, left=0.3, right=0.75, top=0.19, bottom=0.055)
    low = [mid[0], mid[1], low[0]]

    # A: structure
    a = fig.add_subplot(top[0])
    structure(a)
    a.set_title("A  One field over organ × space", loc="left")
    check_inside(fig, a, a.texts, "axes")

    # B: the field at three times, side by side in one image (a blank column between)
    t, C = iv["t"], iv["C"]
    whens = (0.5, 6.0, 48.0)
    blocks = []
    for when in whens:
        blocks += [C[int(np.argmin(np.abs(t - when)))], np.full((9, 1), np.nan)]
    a = fig.add_subplot(top[1])
    im = a.imshow(np.hstack(blocks[:-1]), norm=LogNorm(1e-3, 30), cmap="viridis", aspect="auto")
    a.set_yticks(range(9), pb.ORGANS)
    a.set_xticks([j + 4 * k for k in range(3) for j in range(3)], ["v", "i", "c"] * 3)
    a.tick_params(length=0)
    for k, when in enumerate(whens):
        a.text(4 * k + 1, -0.6, f"{when:g} h", ha="center", va="bottom")
    a.set_title("B  C(organ, space) after 100 mg IV", loc="left", pad=14)
    for s in a.spines.values():
        s.set_visible(False)
    cax = a.inset_axes([1.03, 0.0, 0.035, 1.0])
    fig.colorbar(im, cax=cax, label="mg/L")

    # C: profiles
    a = fig.add_subplot(low[0])
    a.semilogy(t, iv["Cart"], "k", lw=1.0, label="arterial blood")
    for name in ("brain", "muscle", "adipose", "liver"):
        o = pb.ORGANS.index(name)
        (line,) = a.semilogy(t, C[:, o, :] @ pb.FRAC, label=name)
        if name == "liver":
            for s, ls in zip(range(3), (":", "-.", "--"), strict=True):
                a.semilogy(t, C[:, o, s], ls=ls, lw=0.8, c=line.get_color(), label=f"liver, {pb.SPACES[s]}")
    a.set(xlim=(0, 96), ylim=(1e-3, 300), xlabel="Time (h)", ylabel="Concentration (mg/L)")
    a.set_title("C  Profiles", loc="left")
    a.legend(ncol=2, frameon=False, handlelength=1.5, columnspacing=0.8, loc="upper right")

    # D: clearance against permeability
    a = ax_d = fig.add_subplot(low[1])
    ps = np.geomspace(1, 1e5, 200)
    a.semilogx(ps, [pb.hepatic_clearance(p, p) for p in ps], c="grey", label="series formula")
    a.semilogx(cl["ps"], cl["simulated"], "o", ms=3, c="C0", label="TenQSP, steady state")
    a.axhline(pb.well_stirred(), c="k", ls="--", lw=0.8, label="well-stirred limit")
    a.set(xlabel="Permeability PS$_1$ = PS$_2$ (L/h)", ylabel="Hepatic clearance (L/h)")
    a.set_title("D  Hepatic clearance", loc="left")
    a.legend(frameon=False, loc="lower right")

    # E: exact checks
    a = fig.add_subplot(low[2])
    short = {"field = 29 hand-written ODEs": "field = 29 ODEs", "mass balance (mg, of 100)": "mass balance",
             "AUC = Dose / CL_H": "AUC = Dose/CL$_H$", "steady state Css = R / CL_H": "C$_{ss}$ = R/CL$_H$",
             "tissue:plasma = Kp_tissue": "tissue:plasma = K$_p$",
             "CL_H = series formula, PS 3 to 1000": "CL$_H$ = series formula"}
    names = list(checks)
    vals = [max(checks[k], 1e-16) for k in names]
    a.barh(range(len(names)), vals, color="C2", height=0.6)
    a.set_xscale("log")
    a.set_xlim(1e-16, 1e-6)
    a.set_yticks(range(len(names)), [short.get(k, k) for k in names])
    a.invert_yaxis()
    a.set(xlabel="Relative error")
    a.set_title("E  Exact checks", loc="left")

    check_inside(fig, a, [t for ax in fig.axes for t in ax.texts] + a.get_yticklabels(), "figure")
    d_box = ax_d.get_tightbbox(fig.canvas.get_renderer())  # E's labels must not reach into D
    if any(t.get_window_extent().overlaps(d_box) for t in a.get_yticklabels()):
        raise SystemExit("panel E's labels overlap panel D")
    pb.ROOT.joinpath("out").mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(pb.ROOT / "out" / f"fig5_pbpk.{ext}", dpi=DPI)
    print(f"wrote {pb.ROOT / 'out' / 'fig5_pbpk.png'}", file=sys.stderr)


if __name__ == "__main__":
    main()
