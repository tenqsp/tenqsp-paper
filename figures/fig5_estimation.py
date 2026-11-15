# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Figure 5: estimation with TenQSP models (tenqsp-fit). Drawn from results/fig5 (run
fig5_runs.py, fig5_theoph.py and fig5_friberg.py first).

    python figures/fig5_estimation.py      # writes out/fig5_estimation.png and .pdf

A  one simulated study (transit time CV 0.20) and the fits of chains of 3, 25 and 60
   compartments; open markers: samples below the limit of quantification
B  estimates against the chain length, relative to the truth (median and 10-90 % range over
   the simulated studies): short chains bias every parameter
C  -2LL against the transit-time CV of the chain, 1/sqrt(N), relative to the best chain of
   each study: the data reject short chains and pick a CV near the truth's (dotted); with its
   longer tail, the truth sits between the chains of 25 and 40
D  theophylline (12 subjects): -2LL against the chain length; N = 0 is first-order absorption
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
import transit_fit as tr

OUT = tr.ROOT / "out"
SHOWN = {3: "C3", 25: "C0", 60: "C2"}
PARAMS = {"MTT": "C0", "ka": "C1", "CL": "C2", "V": "C3"}
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
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))
    ax = ax.ravel()

    # A
    t = np.geomspace(0.05, 24, 300)
    obs = data[data.BLQ == 0]
    blq = data[data.BLQ == 1]
    ax[0].plot(obs.TIME, obs.DV, ".", c="0.6", ms=4, label="data")
    ax[0].plot(blq.TIME, blq.DV, "o", c="0.6", ms=4, mfc="none", label="below LLOQ")
    ax[0].plot(t, tr.true_profile(list(t)), "k:", label="truth")
    s1 = fits[fits.study == 1].set_index("N")
    for n, c in SHOWN.items():
        ax[0].plot(t, fitted_profile(n, s1.loc[n], t), c=c, label=f"chain of {n}")
    ax[0].axhline(tr.LLOQ, c="0.8", lw=0.8)
    ax[0].set(xscale="log", yscale="log", xlim=(0.2, 30), ylim=(0.02, 10), xlabel="time (h)",
              ylabel="concentration (mg/L)", title="A  one study and its fits")
    ax[0].legend(fontsize=8)

    # B
    ns = np.array(tr.CHAINS)
    for k, c in PARAMS.items():
        r = fits.pivot(index="N", columns="study", values=k).loc[ns] / tr.TRUE[k]
        ax[1].plot(ns, r.median(axis=1), "o-", c=c, ms=4, label=k)
        ax[1].fill_between(ns, r.quantile(0.1, axis=1), r.quantile(0.9, axis=1), color=c, alpha=0.15, lw=0)
    ax[1].axhline(1, c="k", lw=0.8)
    ax[1].axvline(25, c="k", ls=":", lw=0.8)
    ax[1].set(xscale="log", yscale="log", xlabel="chain length N", ylabel="estimate / truth",
              title=f"B  estimates ({studies} studies)")
    ax[1].set_xticks(ns, [str(n) if n != 30 else "" for n in ns])
    ax[1].set_xticks([], minor=True)
    ax[1].set_yticks([0.5, 1, 2, 4], ["0.5", "1", "2", "4"])
    ax[1].set_yticks([], minor=True)
    ax[1].legend(fontsize=8, ncol=2)

    # C
    m2ll = fits.pivot(index="N", columns="study", values="m2ll").loc[ns]
    d = m2ll - m2ll.min(axis=0)
    cv = 1 / np.sqrt(ns)
    ax[2].plot(cv, d, c="0.75", lw=0.7)
    ax[2].plot(cv, d.median(axis=1), "ko-", ms=4, label="median")
    ax[2].axvline(np.sqrt(2 * tr.DISP), c="k", ls=":", lw=0.8, label="truth, CV 0.20")
    best = d.idxmin(axis=0).value_counts()
    ax[2].text(0.97, 0.05, "best chain (studies):\n" + "\n".join(f"N = {n}: {c}" for n, c in best.sort_index().items()),
               transform=ax[2].transAxes, ha="right", va="bottom", fontsize=8)
    ax[2].set_yscale("symlog", linthresh=1)
    ax[2].set_xscale("log")
    ax[2].set_xticks(cv, [f"{c:.2f}" if n in (3, 10, 25, 60) else "" for n, c in zip(ns, cv, strict=True)])
    ax[2].set_xticks([], minor=True)
    ax[2].set_yticks([0, 1, 10, 100], ["0", "1", "10", "100"])
    ax[2].set_ylim(-0.2, 3 * float(d.max().max()))
    ax[2].set(xlabel="transit-time CV, 1/sqrt(N)",
              ylabel="-2LL - best of study", title="C  the data choose the delay")
    ax[2].legend(fontsize=8, loc="upper left")

    # D
    th = theoph.set_index("N")
    ax[3].plot(th.index[1:], th.m2ll.iloc[1:], "ko-", ms=4, label="chain of N")
    ax[3].axhline(th.m2ll.loc[0], c="C3", ls="--", label="first-order absorption")
    ax[3].set(xscale="log", xlabel="chain length N", ylabel="-2LL", title="D  theophylline")
    ax[3].set_xticks(th.index[1:], [str(n) for n in th.index[1:]])
    ax[3].set_xticks([], minor=True)
    ax[3].legend(fontsize=8)

    # E
    y = np.arange(len(fri))[::-1]
    ax[4].errorbar(fri.estimate / fri.true, y, xerr=[(fri.estimate - fri.lo) / fri.true, (fri.hi - fri.estimate) / fri.true],
                   fmt="o", c="k", ms=4, capsize=2)
    ax[4].axvline(1, c="k", lw=0.8)
    ax[4].set_yticks(y, fri.quantity)
    ax[4].set(xlabel="estimate / truth (95 % interval)", title="E  Friberg model, estimates")

    # F
    dose = fdata[fdata.EVID == 1].groupby("ID").AMT.first()
    obs = fdata[fdata.EVID == 0].assign(dose=lambda x: x.ID.map(dose))
    for amt, c in DOSES.items():
        o = obs[obs.dose == amt]
        ax[5].plot(o.TIME / 24, o.DV, ".", c=c, ms=3, alpha=0.6)
        ax[5].plot(ftyp.t / 24, ftyp[f"dose_{amt}"], c=c, label=f"dose {amt}, typical")
    ax[5].set(yscale="log", xlabel="time (days)", ylabel="neutrophils (10$^9$/L)", title="F  Friberg model, fit")
    ax[5].legend(fontsize=8)

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig5_estimation.{ext}", dpi=200)
    print(f"wrote {OUT / 'fig5_estimation.png'} from {studies} studies", file=sys.stderr)


if __name__ == "__main__":
    main()
