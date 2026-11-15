# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Figure 6: a whole-body PBPK model as one field over organs and spaces. Drawn from
results/fig6 (run fig6_runs.py first). The structure diagram (organs x spaces, blood flows)
is drawn separately; this script draws the computed panels.

    python figures/fig6_pbpk.py      # writes out/fig6_pbpk.png and .pdf

B  the field C(organ, space) after a 100 mg IV dose, at three times (mg/L)
C  arterial blood and whole-organ concentrations; dotted: the liver's three spaces
D  hepatic clearance against the membranes' permeability: simulated (dots), the series
   formula (line) and its limit, the well-stirred liver (dashed)
E  the exact checks: relative error of each
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pbpk as pb
from matplotlib.colors import LogNorm


def main():
    iv = np.load(pb.RESULTS / "iv.npz")
    cl = np.load(pb.RESULTS / "clearance.npz")
    checks = json.loads((pb.RESULTS / "checks.json").read_text())
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 6)

    # B: the field at three times, side by side in one image (a blank column between)
    t, C = iv["t"], iv["C"]
    whens = (0.5, 6.0, 48.0)
    blocks = []
    for when in whens:
        blocks += [C[int(np.argmin(np.abs(t - when)))], np.full((9, 1), np.nan)]
    a = fig.add_subplot(gs[0, :3])
    im = a.imshow(np.hstack(blocks[:-1]), norm=LogNorm(1e-3, 30), cmap="viridis", aspect="auto")
    a.set_yticks(range(9), pb.ORGANS, fontsize=8)
    a.set_xticks([j + 4 * k for k in range(3) for j in range(3)], ["v", "i", "c"] * 3, fontsize=8)
    for k, when in enumerate(whens):
        a.text(4 * k + 1, -0.8, f"t = {when:g} h", ha="center", fontsize=9)
    a.set_title("B  C(organ, space) after 100 mg IV (v, i, c: vascular, interstitial, cellular)",
                fontsize=10, pad=18)
    fig.colorbar(im, ax=a, label="mg/L")

    # C: profiles
    a = fig.add_subplot(gs[0, 3:])
    a.semilogy(t, iv["Cart"], "k", lw=2, label="arterial blood")
    for name in ("brain", "muscle", "adipose", "liver"):
        o = pb.ORGANS.index(name)
        (line,) = a.semilogy(t, C[:, o, :] @ pb.FRAC, label=name)
        if name == "liver":
            for s, ls in zip(range(3), (":", "-.", "--"), strict=True):
                a.semilogy(t, C[:, o, s], ls=ls, lw=1, c=line.get_color(), label=f"liver, {pb.SPACES[s]}")
    a.set(xlim=(0, 96), ylim=(1e-3, 30), xlabel="time (h)", ylabel="concentration (mg/L)",
          title="C  100 mg IV")
    a.legend(fontsize=7, ncol=2)

    # D: clearance against permeability
    a = fig.add_subplot(gs[1, :3])
    ps = np.geomspace(1, 1e5, 200)
    a.semilogx(ps, [pb.hepatic_clearance(p, p) for p in ps], c="grey", label="series formula")
    a.semilogx(cl["ps"], cl["simulated"], "o", c="C0", label="TenQSP, steady state")
    a.axhline(pb.well_stirred(), c="k", ls="--", lw=1, label="well-stirred (PS -> infinity)")
    a.set(xlabel="membrane permeability PS1 = PS2 (L/h)", ylabel="hepatic clearance (L/h)",
          title="D  permeability-limited to well-stirred")
    a.legend(fontsize=8)

    # E: exact checks
    a = fig.add_subplot(gs[1, 3:])
    names = list(checks)
    vals = [max(checks[k], 1e-16) for k in names]
    a.barh(range(len(names)), vals, color="C2")
    a.set_xscale("log")
    a.set_xlim(1e-16, 1e-6)
    a.set_yticks(range(len(names)), names, fontsize=8)
    a.invert_yaxis()
    a.set(xlabel="relative error", title="E  exact checks (rtol 1e-10)")

    fig.tight_layout()
    pb.ROOT.joinpath("out").mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(pb.ROOT / "out" / f"fig6_pbpk.{ext}", dpi=200)


if __name__ == "__main__":
    main()
