# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Figure 6: the tumor spheroid case study, drawn from the stored results (seconds), at print
size (printstyle.py).
Run fig6_runs.py and fig6_gradients.py first (or use the results in the repository).

    python figures/fig6_spheroid.py [--quick]     # writes out/fig6_spheroid.png and .pdf

A  tumor burden under three schedules with the same average plasma level (solid: EC50 = 4;
   dotted: linear kill)
B  mean resistance level
C  mean density against resistance at the end, for each kill and schedule; open symbols: the
   same model without the depth axis (with depth, the burden is divided by the volume 4*pi/3)
D  the cells over depth and resistance at the end (a bolus every 21, EC50 = 4)
E  resistance by depth at the end: fast (epsd = 0.1) and slow (epsd = 5) penetration
F  which doses matter: change of the mean burden over the course for a dose given 1 time
   unit later, or 1% larger (reverse-mode derivatives, one run); inset: the burden at the end
   for a dose 1 time unit later (only the last few doses count)
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SPHERE = 4 * np.pi / 3  # with depth, the burden is the amount in the unit sphere; without, a density
import printstyle as ps
import spheroid as sp

COLORS = {"inf": "C0", "7": "C1", "21": "C3"}
MARKERS = {"inf": "o", "7": "s", "21": "^"}


def main(mode):
    d = sp.results_dir(mode)

    def load(tag):
        return np.load(d / f"{tag}.npz")

    nr, nx = sp.GRIDS[mode]
    r = (np.arange(nr) + 0.5) / nr
    x = (np.arange(nx) + 0.5) / nx
    ps.setup()
    fig, ax = plt.subplots(2, 3, figsize=(ps.WIDTH, 4.7))
    ax = ax.ravel()

    for s, label in sp.SCHEDULES.items():
        c = COLORS[s]
        a, z = load(f"spheroid_pk_0.25_{s}_1"), load(f"spheroid_pk_0_{s}_1")
        ax[0].semilogy(a["t"], a["total"], c=c, label=label)
        ax[0].semilogy(z["t"], z["total"], c=c, ls=":", lw=0.8)
        ax[1].plot(a["t"], a["mean_x"], c=c, label=label)
        ax[1].plot(z["t"], z["mean_x"], c=c, ls=":", lw=0.8)
    ax[0].set(xlabel="Time (dimensionless)", ylabel="Tumor burden", title="A  Burden")
    ax[1].set(xlabel="Time (dimensionless)", ylabel="Mean resistance level", title="B  Resistance")
    ax[0].plot([], [], c="0.4", ls=":", lw=0.8, label="linear kill")
    ax[0].legend(loc="center right", bbox_to_anchor=(1.0, 0.45), handlelength=1.6)

    for isat, label, c in (("0", "linear kill", "k"), ("0.25", "EC50 4", "C2"), ("1", "EC50 1", "C4")):
        for s in sp.SCHEDULES:
            a = load(f"spheroid_pk_{isat}_{s}_1")
            ax[2].scatter(a["mean_x"][-1], a["total"][-1] / SPHERE, marker=MARKERS[s], c=c, s=14,
                          label=label if s == "inf" else None)
    for s in sp.SCHEDULES:
        a = load(f"spheroid_nodepth_0.25_{s}")
        ax[2].scatter(a["mean_x"][-1], a["total"][-1], marker=MARKERS[s], facecolors="none", s=14,
                      edgecolors="C2", label="EC50 4, no depth" if s == "inf" else None)
    ax[2].set(yscale="log", xlabel="Resistance at the end", ylabel="Mean density at the end",
              title="C  At the end")
    ax[2].text(0.03, 0.03, "o infusion, □ every 7, △ every 21", transform=ax[2].transAxes, va="bottom",
               fontsize=ps.SIZE - 1)
    ax[2].legend(loc="center left")

    n = load("spheroid_pk_0.25_21_1")["n"][-1]
    im = ax[3].pcolormesh(r, x, np.log10(np.maximum(n, 1e-12)).T, vmin=-8, shading="auto")
    fig.colorbar(im, ax=ax[3], label="log$_{10}$ cell density", fraction=0.07, pad=0.03)
    ax[3].set(xlabel="Distance from the center r", ylabel="Resistance level x",
              title="D  Cells at the end")

    for s, ls in (("7", "--"), ("21", "-")):
        for e, c, speed in (("0.1", "C0", "fast"), ("5", "C3", "slow")):
            n = load(f"spheroid_pk_0.25_{s}_{e}")["n"][-1]
            ax[4].plot(r, (n * x).sum(1) / n.sum(1), c=c, ls=ls, label=f"every {s}, {speed}")
    ax[4].set(xlabel="Distance from the center r", ylabel="Mean resistance level",
              title="E  Resistance by depth")
    ax[4].set_ylim(top=0.885)
    ax[4].legend(ncol=2, loc="upper center", columnspacing=0.8, handlelength=1.6, title="schedule, penetration",
                 title_fontsize=ps.SIZE - 1, fontsize=ps.SIZE - 1)

    g = d / "gradients.npz"
    if g.exists():
        g = np.load(g)
        k = len(g["dose_times"])
        J, y, x0 = g["J"], g["y"], g["x0"]
        later = 100 * J[0, :k] / y[0]  # % change of the mean burden, dose 1 time unit later
        larger = J[0, k:] * x0[k:] / y[0]  # % change for a dose 1 % larger
        number = np.arange(1, k + 1)
        ax[5].bar(number - 0.2, larger, width=0.4, color="C0", label="each dose 1% larger")
        ax[5].bar(number + 0.2, later, width=0.4, color="C3", label="each dose 1 time unit later")
        ax[5].axhline(0, c="k", lw=0.5)
        ax[5].set(xlabel="Dose number (every 21)", ylabel="% change of the mean burden",
                  title="F  Which doses matter")
        ax[5].legend(loc="upper left", fontsize=ps.SIZE - 1)
        # inset: the burden at the end forgets all but the last few dose times
        ins = ax[5].inset_axes([0.42, 0.38, 0.52, 0.32])
        ins.bar(number, 100 * J[2, :k], width=0.6, color="C3")  # % per time unit later
        ins.set_title("final burden, dose 1 later", fontsize=ps.SIZE - 1, loc="center", weight="normal")
        ins.tick_params(labelsize=ps.SIZE - 1.5)
        ins.set_xlabel("dose number", fontsize=ps.SIZE - 1.5, labelpad=1)
        ins.set_ylabel("% change", fontsize=ps.SIZE - 1.5, labelpad=1)
    else:
        ax[5].text(0.5, 0.5, "run fig6_gradients.py", ha="center")
        ax[5].set_axis_off()

    fig.tight_layout(w_pad=1.0, h_pad=1.0)
    ps.check(fig, ax)
    out = sp.ROOT / "out"
    out.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig6_spheroid{'_quick' if mode == 'quick' else ''}.{ext}")
    print("written to", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main("quick" if ap.parse_args().quick else "full")
