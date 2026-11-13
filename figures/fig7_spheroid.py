# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Figure 7: the tumour spheroid case study, drawn from the stored results (seconds).
Run fig7_runs.py and fig7_gradients.py first (or use the results in the repository).

    python figures/fig7_spheroid.py [--quick]     # writes out/fig7_spheroid.png and .pdf

A  tumour burden under three schedules with the same average plasma level (solid: EC50 = 4;
   dotted: linear kill)
B  mean resistance level
C  the trade-off: burden against resistance at the end, for each kill and schedule; open
   symbols: the same model without the depth axis
D  the cells over depth and resistance at the end (a bolus every 21, EC50 = 4)
E  resistance by depth at the end: fast (epsd = 0.1) and slow (epsd = 5) penetration
F  which doses matter: change of the mean burden over the course for a dose given 1 time
   unit later, or 1 % larger (exact derivatives, one run); inset: the burden at the end
   for a dose 1 time unit later (only the last few doses count)
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spheroid as sp

COLOURS = {"inf": "C0", "7": "C1", "21": "C3"}
MARKERS = {"inf": "o", "7": "s", "21": "^"}


def main(mode):
    d = sp.results_dir(mode)

    def load(tag):
        return np.load(d / f"{tag}.npz")

    nr, nx = sp.GRIDS[mode]
    r = (np.arange(nr) + 0.5) / nr
    x = (np.arange(nx) + 0.5) / nx
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))
    ax = ax.ravel()

    for s, label in sp.SCHEDULES.items():
        c = COLOURS[s]
        a, z = load(f"spheroid_pk_0.25_{s}_1"), load(f"spheroid_pk_0_{s}_1")
        ax[0].semilogy(a["t"], a["total"], c=c, label=label)
        ax[0].semilogy(z["t"], z["total"], c=c, ls=":", lw=1)
        ax[1].plot(a["t"], a["mean_x"], c=c, label=label)
        ax[1].plot(z["t"], z["mean_x"], c=c, ls=":", lw=1)
    ax[0].set(xlabel="time", ylabel="tumour burden", title="A  burden")
    ax[1].set(xlabel="time", ylabel="mean resistance level", title="B  resistance")
    ax[0].legend(fontsize=8)

    for isat, label, c in (("0", "linear kill", "k"), ("0.25", "EC50 4", "C2"), ("1", "EC50 1", "C4")):
        for s in sp.SCHEDULES:
            a = load(f"spheroid_pk_{isat}_{s}_1")
            ax[2].scatter(a["mean_x"][-1], a["total"][-1], marker=MARKERS[s], c=c,
                          label=label if s == "inf" else None)
    for s in sp.SCHEDULES:
        a = load(f"spheroid_nodepth_0.25_{s}")
        ax[2].scatter(a["mean_x"][-1], a["total"][-1], marker=MARKERS[s], facecolors="none",
                      edgecolors="C2", label="EC50 4, no depth" if s == "inf" else None)
    ax[2].set(yscale="log", xlabel="resistance at the end", ylabel="burden at the end",
              title="C  trade-off (o inf., □ q7, △ q21)")
    ax[2].legend(fontsize=8)

    n = load("spheroid_pk_0.25_21_1")["n"][-1]
    im = ax[3].pcolormesh(r, x, np.log10(np.maximum(n, 1e-12)).T, vmin=-8, shading="auto")
    fig.colorbar(im, ax=ax[3], label="log10 cell density")
    ax[3].set(xlabel="distance from the centre r", ylabel="resistance level x",
              title="D  cells at the end (every 21, EC50 4)")

    for s, ls in (("7", "--"), ("21", "-")):
        for e, c, speed in (("0.1", "C0", "fast"), ("5", "C3", "slow")):
            n = load(f"spheroid_pk_0.25_{s}_{e}")["n"][-1]
            ax[4].plot(r, (n * x).sum(1) / n.sum(1), c=c, ls=ls,
                       label=f"every {s}, {speed} penetration")
    ax[4].set(xlabel="distance from the centre r", ylabel="mean resistance level",
              title="E  resistance by depth at the end")
    ax[4].legend(fontsize=8)

    g = d / "gradients.npz"
    if g.exists():
        g = np.load(g)
        k = len(g["dose_times"])
        J, y, x0 = g["J"], g["y"], g["x0"]
        later = 100 * J[0, :k] / y[0]  # % change of the mean burden, dose 1 time unit later
        larger = J[0, k:] * x0[k:] / y[0]  # % change for a dose 1 % larger
        number = np.arange(1, k + 1)
        ax[5].bar(number - 0.2, larger, width=0.4, color="C0", label="dose 1 % larger")
        ax[5].bar(number + 0.2, later, width=0.4, color="C3", label="dose 1 time unit later")
        ax[5].axhline(0, c="k", lw=0.5)
        ax[5].set(xlabel="dose number (every 21)", ylabel="% change of the mean burden",
                  title="F  which doses matter")
        ax[5].legend(fontsize=8, loc="upper left")
        # inset: the burden at the end forgets all but the last few dose times
        ins = ax[5].inset_axes([0.38, 0.47, 0.45, 0.3])
        ins.bar(number, 100 * J[2, :k], width=0.6, color="C3")  # % per time unit later
        ins.set_title("burden at the end, dose 1 later", fontsize=7)
        ins.tick_params(labelsize=6)
        ins.set_xlabel("dose number", fontsize=6)
        ins.set_ylabel("% change", fontsize=6)
    else:
        ax[5].text(0.5, 0.5, "run fig7_gradients.py", ha="center")
        ax[5].set_axis_off()

    fig.tight_layout()
    out = sp.ROOT / "out"
    out.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig7_spheroid{'_quick' if mode == 'quick' else ''}.{ext}", dpi=150)
    print("written to", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main("quick" if ap.parse_args().quick else "full")
