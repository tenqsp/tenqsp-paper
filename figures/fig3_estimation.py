# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Figure 3: estimation with TenQSP models (tenqsp-fit). Drawn from results/fig3 (run
fig3_runs.py, fig3_theoph.py and fig3_friberg.py first).

    python figures/fig3_estimation.py      # writes out/fig3_estimation.png and .pdf

Drawn at print size (printstyle.py).

A  one simulated study (transit time CV 0.20) and the fits of chains of 3, 25 and 60
   compartments; open markers: samples below the limit of quantification
B  estimates against the chain length, relative to the truth (median and 10-90 % range over
   the simulated studies): short chains bias every parameter
C  -2LL against the transit-time CV of the chain, 1/sqrt(N), relative to the best chain of
   each study: the data reject short chains and pick a CV near the truth's (dotted); with its
   longer tail, the truth sits between the chains of 25 and 40
D  theophylline (12 subjects): -2LL against the chain length, first-order absorption, and
   transport with an estimated dispersion on 20 and 40 grid cells (when fitted)
E  Friberg's myelosuppression model fitted to a simulated study: estimates with 95 %
   intervals, relative to the truth
F  the same study: neutrophil counts and the fitted typical profiles for each dose
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import printstyle as ps
import transit_fit as tr

OUT = tr.ROOT / "out"
SHOWN = {3: "C3", 25: "C0", 60: "C2"}
PARAMS = {"MTT": "C0", "MTT + 1/ka": "C1", "CL": "C2", "V": "C3"}  # ka alone collapses onto N/MTT for N <= 20
NAMES = {"circ0.median": "Circ$_0$", "slope.median": "Slope", "MTT": "MTT", "gamma": "$\\gamma$",
         "circ0.sdlog": "$\\omega$ Circ$_0$", "slope.sdlog": "$\\omega$ Slope", "ANC.proportional": "$\\sigma$ (proportional)"}
DOSES = {60: "C0", 90: "C1", 120: "C3"}


def fitted_profile(n, row, times):
    m = tr.transit_model(n)
    theta = {k: float(row[k]) for k in tr.TRUE}
    return np.asarray(m.solve((0, times[-1]), times=times, params=theta, rtol=1e-8, atol=1e-10)["conc"])


def main():
    fits = pd.read_csv(tr.RESULTS / "fits.csv")
    data = pd.read_csv(tr.RESULTS / "study1.csv")
    theoph = pd.read_csv(tr.RESULTS / "theoph.csv")
    fri = pd.read_csv(tr.RESULTS / "friberg_estimates.csv")
    fdata = pd.read_csv(tr.RESULTS / "friberg_data.csv")
    ftyp = pd.read_csv(tr.RESULTS / "friberg_typical.csv")
    studies = fits["study"].nunique()
    ps.setup()
    fig, ax = plt.subplots(3, 2, figsize=(ps.WIDTH, 7.8))
    ax = ax.ravel()

    # A
    t = np.geomspace(0.05, 24, 300)
    obs = data[data.BLQ == 0]
    blq = data[data.BLQ == 1]
    ax[0].plot(obs.TIME, obs.DV, ".", c="0.6", ms=2.5, label="data")
    ax[0].plot(blq.TIME, blq.DV, "o", c="0.6", ms=2.5, mfc="none", label="below LLOQ")
    ax[0].plot(t, tr.true_profile(list(t)), "k:", label="truth")
    s1 = fits[fits.study == 1].set_index("N")
    for n, c in SHOWN.items():
        ax[0].plot(t, fitted_profile(n, s1.loc[n], t), c=c, label=f"chain of {n}")
    ax[0].axhline(tr.LLOQ, c="0.8", lw=0.8)
    ax[0].set(xscale="log", yscale="log", xlim=(0.2, 30), ylim=(0.02, 200), xlabel="Time (h)",
              ylabel="Concentration (mg/L)", title="A  One study and its fits")
    ax[0].legend(loc="upper center", ncol=2, columnspacing=0.6, handlelength=1.2)

    # B
    ns = np.array(tr.CHAINS)
    for k, c in PARAMS.items():
        if k == "MTT + 1/ka":
            v = fits.assign(total=fits.MTT + 1 / fits.ka).pivot(index="N", columns="study", values="total")
            r = v.loc[ns] / (tr.TRUE["MTT"] + 1 / tr.TRUE["ka"])
        else:
            r = fits.pivot(index="N", columns="study", values=k).loc[ns] / tr.TRUE[k]
        ax[1].plot(ns, r.median(axis=1), "o-", c=c, ms=2.5, label=k)
        ax[1].fill_between(ns, r.quantile(0.1, axis=1), r.quantile(0.9, axis=1), color=c, alpha=0.15, lw=0)
    ax[1].axhline(1, c="k", lw=0.8)
    ax[1].axvline(25, c="k", ls=":", lw=0.8)
    ax[1].set(xscale="log", yscale="log", xlabel="Chain length N", ylabel="Estimate / truth",
              title=f"B  Estimates ({studies} studies)")
    ax[1].set_xticks(ns, [str(n) if n not in (20, 30) else "" for n in ns])
    ax[1].set_xticks([], minor=True)
    ax[1].set_yticks([0.5, 1, 2, 4], ["0.5", "1", "2", "4"])
    ax[1].set_yticks([], minor=True)
    ax[1].legend(ncol=2, loc="upper left", columnspacing=0.8, handlelength=1.4)

    # C
    m2ll = fits.pivot(index="N", columns="study", values="m2ll").loc[ns]
    d = m2ll - m2ll.min(axis=0)
    cv = 1 / np.sqrt(ns)
    ax[2].plot(cv, d, c="0.75", lw=0.7)
    ax[2].plot(cv, d.median(axis=1), "ko-", ms=2.5, label="median")
    ax[2].axvline(np.sqrt(2 * tr.DISP), c="k", ls=":", lw=0.8, label="truth, CV 0.20")
    best = d.idxmin(axis=0).value_counts()
    ax[2].text(0.97, 0.05, "best chain (studies):\n" + "\n".join(f"N = {n}: {c}" for n, c in best.sort_index().items()),
               transform=ax[2].transAxes, ha="right", va="bottom", fontsize=ps.SIZE)
    ax[2].set_yscale("symlog", linthresh=1)
    ax[2].set_xscale("log")
    ax[2].set_xticks(cv, [f"{c:.2f}" if n in (3, 10, 25, 60) else "" for n, c in zip(ns, cv, strict=True)])
    ax[2].set_xticks([], minor=True)
    ax[2].set_yticks([0, 1, 10, 100], ["0", "1", "10", "100"])
    ax[2].set_ylim(-0.2, 3 * float(d.max().max()))
    ax[2].set(xlabel="Transit-time CV, 1/$\\sqrt{N}$",
              ylabel="$-$2LL $-$ best of study", title="C  $-$2LL against chain length")
    ax[2].legend(loc="upper right", bbox_to_anchor=(1.0, 0.66))  # clear of the study lines and the truth

    # D
    th = theoph.set_index("N")
    ax[3].plot(th.index[1:], th.m2ll.iloc[1:], "ko-", ms=2.5, label="chain of N")
    ax[3].axhline(th.m2ll.loc[0], c="C3", ls="--", label="first-order absorption")
    disp = tr.RESULTS / "theoph_dispersion.csv"
    if disp.exists():
        for (_, row), ls in zip(pd.read_csv(disp).iterrows(), ("-", ":"), strict=False):
            ax[3].axhline(row.m2ll, c="C0", ls=ls, lw=0.9, label=f"dispersion, {int(row.N)} grid cells")
    ax[3].set(xscale="log", xlabel="Chain length N", ylabel="$-$2LL", title="D  Theophylline")
    ax[3].set_xticks(th.index[1:], [str(n) for n in th.index[1:]])
    ax[3].set_xticks([], minor=True)
    ax[3].legend(loc="center right")

    # E
    y = np.arange(len(fri))[::-1]
    ax[4].errorbar(fri.estimate / fri.true, y, xerr=[(fri.estimate - fri.lo) / fri.true, (fri.hi - fri.estimate) / fri.true],
                   fmt="o", c="k", ms=2.5, capsize=1.5, lw=0.8)
    ax[4].axvline(1, c="k", lw=0.8)
    ax[4].set_yticks(y, [NAMES.get(q, q) for q in fri.quantity])
    ax[4].set(xlabel="Estimate / truth (95% interval)", title="E  Friberg model, estimates")

    # F
    dose = fdata[fdata.EVID == 1].groupby("ID").AMT.first()
    obs = fdata[fdata.EVID == 0].assign(dose=lambda x: x.ID.map(dose))
    for amt, c in DOSES.items():
        o = obs[obs.dose == amt]
        ax[5].plot(o.TIME / 24, o.DV, ".", c=c, ms=2, alpha=0.6)
        ax[5].plot(ftyp.t / 24, ftyp[f"dose_{amt}"], c=c, label=f"{amt} mg")
    ax[5].set(yscale="log", xlabel="Time (days)", ylabel="Neutrophils (10$^9$/L)", title="F  Friberg model, fit")
    ax[5].set_ylim(top=20)
    ax[5].legend(loc="lower right", handlelength=1.2)

    fig.tight_layout(w_pad=1.0, h_pad=1.0)
    ps.check(fig, ax)
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig3_estimation.{ext}")
    print(f"wrote {OUT / 'fig3_estimation.png'} from {studies} studies", file=sys.stderr)


if __name__ == "__main__":
    main()
