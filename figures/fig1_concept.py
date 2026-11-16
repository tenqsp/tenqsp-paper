# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Figure 1: from equations to estimates, on Friberg's myelosuppression model with the
maturation chain written as an axis (models/friberg_maturation.tq). Everything shown is
TenQSP's own output for that file; panel F is the fit of Figure 5 (run fig5_friberg.py
first).

    python figures/fig1_concept.py     # writes out/fig1_concept.png and .pdf (under a minute)

Drawn at the journal's full width (7 in), text at 6-7 pt; it stops if any text leaves its box.

A  the model as written (.tq; comments left out)
B  the equations TenQSP reads from it (m.to_latex(), shown here with matplotlib's mathtext)
C  what TenQSP makes of them (m.explain(), excerpts)
D  a run after a 90 mg dose: proliferating cells, the 3 maturation cells, circulating
   neutrophils; and the conservation ledger's largest unexplained part
E  exact derivatives of the neutrophil count with respect to the parameters (JAX, forward
   mode), as relative sensitivities d log ANC / d log theta
F  a population fit with tenqsp-fit (laplace, FOCE-I; the code is in fig5_friberg.py): 20
   simulated patients at 3 doses, the typical profiles at the estimates
"""

import re
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tenqsp as tq
from transit_fit import RESULTS, ROOT

jax.config.update("jax_enable_x64", True)

OUT = ROOT / "out"
MODEL = ROOT / "models" / "friberg_maturation.tq"
DOSE = 90.0
WRT = ["MTT", "gamma", "circ0", "slope"]
WIDTH = 50  # characters per line in panel C
DPI = 300
SIZE = 6.5  # points, at print size: the figure is drawn at the journal's full width
EXCERPTS = [  # panel C: pieces of m.explain(), verbatim, in its order ("..." marks a cut)
    ("transport along z", "transit compartments"),
    ("Mean transit time", "= 0.577."),
    ("Terms shared", "Transit leaving at z=1 -1."),
    ("No value is clipped", "reported by diagnostics(sol)"),
]


def model_text():
    lines = [ln for ln in MODEL.read_text().splitlines() if not ln.startswith("#")]
    return "\n".join(lines)


def mathtext_lines(latex):
    """m.to_latex()'s align environment as mathtext, one entry per heading or line: the
    equations in display size, the boundaries a line each, and the short definitions under
    "where" and "starting values" two to a line."""
    body = latex.split(r"\begin{align*}")[1].split(r"\end{align*}")[0]
    out, section, pending = [], "equations", False
    for raw in body.split(r"\\"):
        raw = raw.strip()
        m = re.match(r"\\intertext\{([^}]*)\}(.*)", raw)
        if m:
            section = m.group(1).split(" (")[0]
            out.append(("text", section))
            raw, pending = m.group(2).strip(), False
        raw = raw.replace(r"\big|", "|").replace("&", "")
        raw = re.sub(r"\\text\{([^}]*)\}", lambda g: r"\mathrm{" + g.group(1).replace(" ", r"\ ") + "}", raw)
        if section == "equations":
            out.append(("math", "$" + raw.replace(r"\frac", r"\dfrac", 1) + "$"))
        elif section == "boundaries" or not pending:
            out.append(("math", f"${raw}$"))
            pending = section != "boundaries"
        else:
            out[-1] = ("math", out[-1][1][:-1] + r",\quad " + raw + "$")
            pending = False
    return out


def explain_excerpt(text):
    """The EXCERPTS of m.explain(), with its own line breaks undone and wrapped to WIDTH."""
    flat = re.sub(r"\s+", " ", text)
    out = []
    for first, last in EXCERPTS:
        i = flat.index(first)
        piece = flat[i:flat.index(last, i) + len(last)]
        if first == "Mean transit time":
            out[-1] += " " + piece
        elif first == "Terms shared":  # a heading and one term per line, as explain() prints them
            head, *terms = re.split(r" (?=\S+: )", piece)
            out += [head, *("  " + t for t in terms)]
        else:
            out.append(piece + ("" if last.endswith((".", ")")) else " ..."))
    lines = text.splitlines()[:3]
    for para in out:
        if not para.startswith(" "):
            lines.append("")
        indent = " " * (len(para) - len(para.lstrip()))
        lines += textwrap.wrap(para.strip(), WIDTH, initial_indent=indent, subsequent_indent=indent + "  ")
    return "\n".join(lines)


def runs():
    m = tq.Model.from_text(MODEL.read_text())
    t = np.linspace(0, 504, 253)
    reg = tq.bolus("Ac", DOSE, times=[0])
    sol = m.solve((0, 504), times=t, regimen=reg, rtol=1e-8, atol=1e-10)
    sim = m.simulator(outputs=["ANC"], backend="jax", rtol=1e-8, atol=1e-10)
    days = np.arange(0, 22, 1.0)
    f = sim.function(tq.Scenario(regimen=reg, times=24 * days), wrt=WRT)
    x0 = jnp.asarray(f.x0)
    jac = np.asarray(jax.jacfwd(lambda x: f(x)["ANC"])(x0))
    anc = np.asarray(f(x0)["ANC"])
    rel = jac * np.asarray(x0)[None, :] / anc[:, None]
    return m, sol, days, rel, f.derivatives


def text_panel(ax, title, body, size=SIZE, family="monospace"):
    ax.axis("off")
    ax.set_title(title, loc="left")
    box(ax)
    return ax.text(0.03, 0.97, body, va="top", ha="left", family=family, fontsize=size, transform=ax.transAxes,
                   linespacing=1.25)


def box(ax):
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, ec="0.75", lw=0.6, clip_on=False))


def check_fits(fig, ax, texts):
    """Stop if any text runs out of its panel's box (it would be cut or overlap at print)."""
    fig.canvas.draw()
    inner = ax.get_window_extent()
    for t in texts:
        bb = t.get_window_extent()
        if bb.x1 > inner.x1 or bb.y0 < inner.y0:
            raise SystemExit(f"text runs out of panel {ax.get_title(loc='left')!r}: right {bb.x1 - inner.x1:.0f}, bottom {inner.y0 - bb.y0:.0f} px: {t.get_text()[:40]!r}")


def main():
    m, sol, days, rel, mode = runs()
    plt.rcParams.update({"font.size": SIZE, "axes.titlesize": 7.5, "axes.titleweight": "bold",
                         "axes.labelsize": SIZE + 0.5, "legend.fontsize": SIZE - 0.5, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.1})
    fig = plt.figure(figsize=(7.0, 8.3), dpi=DPI)  # the checks below measure at print resolution
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1.1, 1], width_ratios=[1, 1], hspace=0.27, wspace=0.22,
                          left=0.075, right=0.99, top=0.965, bottom=0.05)

    # A
    ax = fig.add_subplot(gs[0, 0])
    check_fits(fig, ax, [text_panel(ax, "A  The model, as written (.tq file)", model_text(), size=SIZE - 0.3)])

    # B
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.set_title("B  The equations read: m.to_latex()", loc="left")
    box(ax)
    pt = fig.dpi / 72 / ax.get_window_extent().height  # one point, in axes units
    y, texts = 0.97, []
    for kind, s in mathtext_lines(m.to_latex()):
        if kind == "text":
            y -= 3 * pt
            texts.append(ax.text(0.04, y, s, style="italic", color="0.35", va="top", transform=ax.transAxes))
            y -= 9 * pt
        else:
            texts.append(ax.text(0.06, y, s, fontsize=SIZE + 0.5, va="top", transform=ax.transAxes))
            y -= (17 if r"\dfrac" in s else 11) * pt
    check_fits(fig, ax, texts)

    # C
    ax = fig.add_subplot(gs[1, 0])
    check_fits(fig, ax, [text_panel(ax, "C  What TenQSP says of them: m.explain()", explain_excerpt(m.explain()),
                                    size=SIZE - 0.3)])

    # D
    ax = fig.add_subplot(gs[1, 1])
    td = sol.t / 24
    ax.plot(td, sol["Prol"], c="C0", label="Prol")
    width = 1 / 3
    for i in range(3):
        ax.plot(td, np.asarray(sol["Transit"])[:, i] * width, c="C1", lw=0.8, alpha=0.5 + 0.25 * i,
                label="Transit (3 cells)" if i == 0 else None)
    ax.plot(td, sol["Circ"], c="C3", lw=1.6, label="Circ = ANC")
    worst = max(sol.ledger.check().values())
    ax.text(0.03, 0.84, f"conservation ledger:\nlargest unexplained part {worst:.0e}", transform=ax.transAxes,
            ha="left", va="top")
    ax.set_ylim(0, 9.5)
    ax.set(xlabel="Time (days)", ylabel="Cells (10$^9$/L)")
    ax.set_title(f"D  A run: m.solve(), {DOSE:.0f} mg at day 0", loc="left")
    ax.legend(loc="upper right", ncol=3, frameon=False, handlelength=1.5, columnspacing=1)

    # E
    ax = fig.add_subplot(gs[2, 0])
    for j, name in enumerate(WRT):
        ax.plot(days, rel[:, j], "o-", ms=2, lw=0.9, label=name)
    ax.axhline(0, c="k", lw=0.5)
    ax.set(xlabel="Time (days)", ylabel="d log ANC / d log $\\theta$")
    ax.set_title(f"E  Exact derivatives (JAX, {mode} mode)", loc="left")
    ax.legend(ncol=2, loc="lower left", frameon=False)

    # F
    ax = fig.add_subplot(gs[2, 1])
    fdata = pd.read_csv(RESULTS / "friberg_data.csv")
    ftyp = pd.read_csv(RESULTS / "friberg_typical.csv")
    dose = fdata[fdata.EVID == 1].groupby("ID").AMT.first()
    obs = fdata[fdata.EVID == 0].assign(dose=lambda x: x.ID.map(dose))
    for amt, c in {60: "C0", 90: "C1", 120: "C3"}.items():
        o = obs[obs.dose == amt]
        ax.plot(o.TIME / 24, o.DV, ".", c=c, ms=2.5, alpha=0.6)
        ax.plot(ftyp.t / 24, ftyp[f"dose_{amt}"], c=c, label=f"{amt} mg")
    ax.set(yscale="log", xlabel="Time (days)", ylabel="ANC (10$^9$/L)", ylim=(0.2, 15))
    ax.set_title("F  A population fit (tenqsp-fit, FOCE-I)", loc="left")
    ax.legend(loc="lower right", frameon=False)

    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig1_concept.{ext}", dpi=DPI)
    print(f"wrote {OUT / 'fig1_concept.png'} (TenQSP {tq.__version__})", file=sys.stderr)


if __name__ == "__main__":
    main()
