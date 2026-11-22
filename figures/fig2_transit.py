# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Figure 2: transit chains are discretized transport, and their length changes how fast an
infection takes off but not its threshold. Drawn from results/fig2 (run fig2_runs.py first).

    python figures/fig2_transit.py      # writes out/fig2_transit.png and .pdf

Drawn at print size (printstyle.py).

A  Savic's transit absorption: a chain of n compartments (lines) and TenQSP's axis with n
   cells (dots) give the same concentrations; inset: the largest difference
B  the eclipse duration of the influenza model (mean 6 h): Erlang(N) and a fixed delay
C  early growth rate of the viral load against N: the characteristic equation (line), the
   leading eigenvalue of TenQSP's Jacobian (dots) and the slope of simulations (crosses);
   dashed: the fixed delay (characteristic root; cross: the sharp scheme)
D  growth rate against the number of target cells: every eclipse shape crosses 0 at the same
   threshold T0 = c delta / (p beta) (R0 = 1); dots: Jacobian eigenvalues
E  viral load with Baccam's parameters: the same R0, but with a more dispersed eclipse the
   infection takes off faster and peaks earlier (N = 1: 0.8 day before the fixed delay)
F  the target cells
"""

import matplotlib

matplotlib.use("Agg")
import eclipse as ec
import matplotlib.pyplot as plt
import numpy as np
import printstyle as ps
from scipy.stats import gamma

COLORS = {1: "C3", 5: "C1", 40: "C0", "delay": "k"}
LABELS = {1: "N = 1 (Baccam)", 5: "N = 5", 40: "N = 40", "delay": "fixed delay"}


def main():
    tr = np.load(ec.RESULTS / "transit.npz")
    r = np.load(ec.RESULTS / "eclipse.npz")
    ps.setup()
    fig, ax = plt.subplots(2, 3, figsize=(ps.WIDTH, 4.6))
    ax = ax.ravel()

    # A
    t = tr["t"]
    diffs = []
    for n, c in ((4, "C0"), (10, "C2")):
        ax[0].plot(t, tr[f"chain_{n}"], c=c, label=f"chain of {n}")
        ax[0].plot(t[::8], tr[f"axis_{n}"][::8], "o", c=c, ms=3, mfc="none", label=f"axis, {n} cells")
        diffs.append(np.max(np.abs(tr[f"axis_{n}"] - tr[f"chain_{n}"])))
    ax[0].set(xlabel="Time (h)", ylabel="Concentration (mg/L)", title="A  Transit absorption")
    ax[0].set_ylim(-0.3, 8.6)
    ax[0].legend(loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.4)
    ax[0].text(0.95, 0.6, f"largest difference\n{max(diffs):.0e} mg/L", transform=ax[0].transAxes,
               ha="right")

    # B
    h = np.linspace(0, 18, 400)  # hours
    for n in (1, 5, 40):
        ax[1].plot(h, gamma.pdf(h, n, scale=6.0 / n), c=COLORS[n], label=LABELS[n])
    ax[1].axvline(6.0, c="k", label="fixed delay")
    ax[1].set(xlabel="Eclipse duration (h)", ylabel="Density (1/h)", title="B  Eclipse, mean 6 h")
    ax[1].legend()

    # C
    ns = r["ns"]
    nn = np.unique(np.round(np.geomspace(1, 60, 80)).astype(int))
    ax[2].plot(nn, [ec.growth_rate(int(n)) for n in nn], c="grey", label="characteristic equation")
    ax[2].plot(ns, r["eig"], "o", c="C0", label="TenQSP Jacobian")
    ax[2].plot(ns, r["sim"], "x", c="k", ms=5, label="TenQSP simulation")
    ax[2].axhline(r["root_delay"], c="k", ls="--", lw=1, label="fixed delay")
    ax[2].plot([60], [r["sim_delay"]], "x", c="k", ms=5)
    ax[2].set_xscale("log")
    ax[2].set_xlim(0.8, 80)
    ax[2].set(xlabel="Eclipse compartments N", ylabel="Early growth rate (1/day)",
              title="C  Growth rate against N")
    ax[2].legend()

    # D
    x = r["t0s"] / ec.T_STAR
    for k in (1, 5, 40, "delay"):
        ax[3].plot(x, r[f"r_vs_T0_{k}"], c=COLORS[k], label=LABELS[k])
    for k in (1, 5, 40):
        ax[3].plot(r["t0_marks"] / ec.T_STAR, r[f"eig_vs_T0_{k}"], "o", c=COLORS[k], ms=3)
    ax[3].axhline(0, c="grey", lw=0.8)
    ax[3].axvline(1, c="grey", lw=0.8, ls=":")
    ax[3].set_xscale("log")
    ax[3].set(xlabel="Target cells / threshold, T$_0$/T* (= R$_0$)", ylabel="Early growth rate (1/day)",
              title="D  Same threshold, R$_0$ = 1")
    ax[3].legend()

    # E
    for k in (1, 5, 40, "delay"):
        ax[4].semilogy(r["t"], r[f"V_{k}"], c=COLORS[k], label=LABELS[k])
        ax[5].plot(r["t"], r[f"T_{k}"] / ec.T0, c=COLORS[k], label=LABELS[k])
    ax[4].set(xlabel="Time (days)", ylabel="Virus (TCID$_{50}$/mL)", title="E  Viral load", ylim=(1e-2, 1e7))
    ax[5].set(xlabel="Time (days)", ylabel="Target cells / T$_0$", title="F  Target cells")
    ax[4].legend()

    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    ps.check(fig, ax)
    ec.ROOT.joinpath("out").mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(ec.ROOT / "out" / f"fig2_transit.{ext}")
    for k in (1, 5, 40, "delay"):
        i = np.argmax(r[f"V_{k}"])
        print(f"{LABELS[k]:16s} peak {r[f'V_{k}'][i]:.3g} TCID50/mL at {r['t'][i]:.2f} days")


if __name__ == "__main__":
    main()
