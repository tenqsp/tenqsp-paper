# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Figure S1: the 12 reference models (TenQSP's tutorials), each as a schematic of its state
and a small view of its solution. Drawn from results/figS1 (run figS1_runs.py first).

    python figures/figS1_gallery.py         # writes out/figS1_gallery.png and .pdf

The schematics are drawn from the models' descriptions, not by hand: a field is a block of
cells, one direction per continuous axis (an arrow for a one-way axis such as an age,
a double arrow for a two-way axis such as depth or a trait), stacked copies for a
categorical axis; circles are compartments; a triangle marks a dose. An arrow from one
part to another means that the second's equations use the first: a flux in or out, or a
value read (a drug concentration, a feedback). Below each: the field's amount over its
first continuous axis (or its categories) and time, from the tutorial's first run of that
model (the gut: the small intestine; the plaque: its finest run, and the phage over space).
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import printstyle as ps
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "figS1"
OUT = ROOT / "out"

# tutorial -> (model, title, reference); VIEW: which run and which field and axis to show below
# (default: the first run, the largest field, its first continuous axis)
MAIN = {
    "01": ("transit_axis", "Transit compartments", "Savic et al. 2007"),
    "02": ("gut", "Gut absorption", "Willmann et al. 2003"),
    "03": ("friberg", "Myelosuppression", "Friberg et al. 2002"),
    "04": ("lifespan_drug", "Lifespan indirect response", "Krzyzanski et al. 1999"),
    "05": ("liver_bolus", "Hepatic dispersion", "Roberts & Rowland 1986"),
    "06": ("pbpk", "Whole-body PBPK", "Jones & Rowland-Yeo 2013"),
    "07": ("hiv_age", "HIV infection age", "Nelson et al. 2004"),
    "08": ("mckendrick", "Age-structured population", "McKendrick 1926"),
    "09": ("wells", "Dormancy, Fokker-Planck", "Nielsen et al. 2007"),
    "10": ("biofilm_bulk", "Biofilm penetration", "Stewart 1996"),
    "11": ("plaque", "Phage plaque fronts", "Yin & McCaskill 1992"),
    "12": ("drug_tolerance", "Drug-tolerance continuum", "Lorenzi et al. 2016"),
}
VIEW = {"02": {"field": "S", "axis": "x"}, "11": {"run": "finest", "field": "P", "axis": "x"}}
MAX_CELLS = 7
FIELD_C = "#cfe0f5"
EDGE = "#2a78d6"
COMP_C = "#fbe3d3"
COMP_E = "#eb6834"


def mentions(expr, name):
    return re.search(rf"(?<![\w.]){re.escape(name)}(?!\w)", str(expr)) is not None


def sizes(ir):
    return {a["name"]: (len(a["faces"]) - 1 if a["type"] == "continuous" else len(a["labels"])) for a in ir["axes"]}


def expand(ir, expr):
    """The expression with definitions and integrals written out (for finding what it reads)."""
    names = {d["name"]: d["expr"] for d in ir["definitions"]}
    names.update({i["name"]: i["integrand"] for i in ir["integrals"]})
    text = str(expr)
    for _ in range(4):
        for k, v in names.items():
            if mentions(text, k):
                text = re.sub(rf"(?<![\w.]){re.escape(k)}(?!\w)", f"({v})", text)
    return text


def links(ir):
    """(source, destination, where) couplings between fields and compartments."""
    fields = [f["name"] for f in ir["fields"]]
    comps = [c["name"] for c in ir["compartments"]]
    out = set()
    for b in ir["boundaries"]:
        for c in comps:
            if b["kind"] in ("flux", "inflow", "dirichlet", "robin") and mentions(expand(ir, b["value"]), c):
                out.add((c, b["target"]))
        if b["kind"] == "outflow" and b.get("to"):
            out.add((b["target"], b["to"]))
    for t in ir["transfers"]:
        if t.get("to"):
            out.add((t["source"], t["to"]))
    for term in ir["terms"]:
        text = expand(ir, term["expr"])
        for other in fields + comps:
            if other != term["target"] and mentions(text, other):
                out.add((other, term["target"]))
    for b in ir["boundaries"]:
        for f in fields:
            if f != b["target"] and mentions(expand(ir, b["value"]), f):
                out.add((f, b["target"]))
    return out


def bend(s, d, lk):
    """Curve an arrow that goes both ways, and one between two compartments (around the field)."""
    if (d, s) in lk:
        return 0.15
    return 0.0


def draw_field(ax, ir, f, x0, y0, w, h):
    """A field as a block of cells; returns its bounding box."""
    n = sizes(ir)
    axes = {a["name"]: a for a in ir["axes"]}
    cont = [a for a in f["axes"] if axes[a]["type"] == "continuous"]
    cat = [a for a in f["axes"] if axes[a]["type"] == "categorical"]
    layers = int(np.prod([n[a] for a in cat])) if cont else 1
    shown_layers = min(layers, 3)
    dx = dy = 0.12 if shown_layers > 1 else 0
    if len(cont) >= 2:
        cols, rows = min(n[cont[0]], MAX_CELLS), min(n[cont[1]], 4)
    elif len(cont) == 1:
        cols, rows = min(n[cont[0]], MAX_CELLS), 1
    else:  # categorical only: first axis across, second down
        cols = min(n[cat[0]], 9)
        rows = min(n[cat[1]], 4) if len(cat) > 1 else 1
    cw = (w - dx * (shown_layers - 1)) / cols
    ch = min((h - dy * (shown_layers - 1)) / rows, min(cw * 1.6, 0.9) if rows == 1 else cw)
    for k in reversed(range(shown_layers)):
        ox, oy = x0 + k * dx, y0 + k * dy
        for i in range(cols):
            for j in range(rows):
                ax.add_patch(Rectangle((ox + i * cw, oy + j * ch), cw, ch, fc=FIELD_C, ec=EDGE, lw=0.6,
                                       alpha=1 if k == 0 else 0.8))
    W, H = cols * cw, rows * ch
    ax.text(x0 + W / 2, y0 + H / 2, f["name"], ha="center", va="center", fontsize=ps.SIZE, weight="bold")
    # axis directions
    def arrow(p, q, kind, label, rot=0):
        style = "-|>" if kind == "one_way" else "<|-|>"
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=6, lw=0.8, color="0.25"))
        mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        if rot == 0:  # at the start, clear of the arrows between fields at the center
            ax.text(p[0] + 0.05, p[1] - 0.14, label, fontsize=5.5, ha="left", va="top")
        elif q[1] - p[1] < 0.8:  # a short block: the label beside it, horizontal
            ax.text(x0 + W + 0.1, mid[1], label, fontsize=5.5, ha="left", va="center")
        else:
            ax.text(mid[0] - 0.12, mid[1], label, fontsize=5.5, ha="right", va="center", rotation=90)

    def axis_label(a):
        g = axes[a].get("geometry", "planar")
        return f"{a}, {n[a]} cells" + ("" if g == "planar" else f" ({g})")

    if cont:
        arrow((x0, y0 - 0.12), (x0 + W, y0 - 0.12), axes[cont[0]]["kind"], axis_label(cont[0]))
    if len(cont) >= 2:
        arrow((x0 - 0.12, y0), (x0 - 0.12, y0 + H), axes[cont[1]]["kind"], axis_label(cont[1]), rot=90)
    if cat and cont:
        labels = [", ".join(axes[a]["labels"][:4]) + (", ..." if n[a] > 4 else "") for a in cat]
        ax.text(x0 + W + dx * (shown_layers - 1) + 0.05, y0 + H + dy * (shown_layers - 1),
                "\n".join(f"{a}: {lab}" for a, lab in zip(cat, labels, strict=True)), fontsize=5.5, va="top")
    elif cat:
        ax.text(x0 + W / 2, y0 - 0.05, f"{cat[0]} ({n[cat[0]]})", fontsize=5.5, ha="center", va="top")
        if len(cat) > 1:
            ax.text(x0 + W + 0.05, y0 + H / 2, f"{cat[1]} ({n[cat[1]]})", fontsize=5.5, ha="left", va="center")
    return (x0, y0, W + dx * (shown_layers - 1), H + dy * (shown_layers - 1))


def schematic(ax, ir):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.1)
    ax.axis("off")
    fields = ir["fields"]
    comps = [c["name"] for c in ir["compartments"]]
    lk = links(ir)
    boxes = {}
    nf = len(fields)
    span, base, gap = (3.0, 0.55, 0.55) if nf < 3 else (3.55, 0.3, 0.7)  # room for each field's labels
    h = span / max(nf, 1)
    for k, f in enumerate(fields):
        boxes[f["name"]] = draw_field(ax, ir, f, 2.6, base + (nf - 1 - k) * h + 0.1, 4.6, h - gap)
    # compartments: sources on the left, sinks on the right, the rest above
    field_names = set(boxes)
    left = [c for c in comps if any(s == c and d in field_names for s, d in lk)]
    right = [c for c in comps if c not in left and any(d == c and s in field_names for s, d in lk)]
    top = [c for c in comps if c not in left and c not in right]
    pos = {}
    for group, x in ((left, 0.9), (right, 8.7)):
        for i, c in enumerate(group):
            pos[c] = (x, 2.9 - i * (2.4 / max(len(group) - 1, 1)) - (0 if len(group) > 1 else 0.8))
    for i, c in enumerate(top):
        pos[c] = (2.6 + i * 1.0, 3.65) if left or right else (1.2, 3.2 - i * 0.9)
    r = 0.38
    for c, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), r, fc=COMP_C, ec=COMP_E, lw=0.8))
        ax.text(x, y, c, ha="center", va="center", fontsize=5.5 if len(c) < 5 else 5)

    def anchor(name, toward):
        if name in pos:
            x, y = pos[name]
            tx, ty = toward
            d = np.hypot(tx - x, ty - y) or 1
            return (x + r * (tx - x) / d, y + r * (ty - y) / d)
        x0, y0, W, H = boxes[name]
        tx, ty = toward
        cx, cy = x0 + W / 2, y0 + H / 2
        if abs(tx - cx) > W / 2:
            return (x0 if tx < cx else x0 + W, cy)
        return (cx, y0 if ty < cy else y0 + H)

    def center(name):
        if name in pos:
            return pos[name]
        x0, y0, W, H = boxes[name]
        return (x0 + W / 2, y0 + H / 2)

    for s, d in sorted(lk):
        p, q = anchor(s, center(d)), anchor(d, center(s))
        if s in boxes and d in boxes:  # between stacked fields: right of center, clear of the axis labels
            def below(name, y):  # an end at a block's bottom starts under its axis arrow
                return y - 0.2 if abs(y - boxes[name][1]) < 1e-9 else y
            p = (boxes[s][0] + 0.85 * boxes[s][2], below(s, p[1]))
            q = (boxes[d][0] + 0.85 * boxes[d][2], below(d, q[1]))
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=COMP_E,
                                     connectionstyle=f"arc3,rad={bend(s, d, lk)}"))
    for dose in ir["doses"]:
        x, y = center(dose["target"])
        top_y = y + (r if dose["target"] in pos else boxes[dose["target"]][3] / 2)
        ax.plot([x], [top_y + 0.22], "v", ms=4, c="k")


def thumbnail(ax, ir, run, view):
    fields = [f for f in ir["fields"] if f["name"] in run["profiles"]]
    f = (next(f for f in fields if f["name"] == view["field"]) if "field" in view
         else max(fields, key=lambda f: run["size_of"].get(f["name"], 0)))
    prof = run["profiles"][f["name"]]
    label = view.get("axis") or next(iter(prof))
    u = np.asarray(prof[label]["amount"], float)
    t = np.asarray(run["t"])
    if label == "categories":
        y = np.arange(u.shape[1] + 1) - 0.5
        label = " x ".join(f["axes"])
    else:
        y = np.asarray(prof[label]["faces"])
    top = u.max()
    positive = u[u > top * 1e-4]
    if top / np.median(positive) > 30:  # wide range: log colors, from 1e-4 of the largest value
        norm, u = LogNorm(vmin=top * 1e-4, vmax=top), np.clip(u, top * 1e-4, None)
    else:
        norm, u = Normalize(0, top), np.clip(u, 0, None)
    tf = np.concatenate([t[:1], (t[1:] + t[:-1]) / 2, t[-1:]])
    ax.pcolormesh(tf, y, u.T, norm=norm, cmap="viridis", shading="flat", rasterized=True)
    ax.set_xlabel("Time", fontsize=ps.SIZE - 0.5, labelpad=1)
    ax.set_ylabel(f"{f['name']}: {label}", fontsize=ps.SIZE - 0.5, labelpad=1)
    ax.tick_params(labelsize=5.5, length=2, pad=1)


def main():
    ps.setup()
    fig = plt.figure(figsize=(ps.WIDTH, 9.4))
    outer = fig.add_gridspec(4, 3, hspace=0.38, wspace=0.36, left=0.065, right=0.978, top=0.965, bottom=0.035)
    panels = []
    version = None
    for k, (key, (name, title, ref)) in enumerate(MAIN.items()):
        p = RESULTS / f"{key}.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        version = data["tenqsp"]
        view = VIEW.get(key, {})
        run = data["models"][name][view.get("run", "first")]
        ir = run["ir"]
        run["size_of"] = {f["name"]: int(np.prod([sizes(ir)[a] for a in f["axes"]])) for f in ir["fields"]}
        cell = outer[k // 3, k % 3].subgridspec(2, 1, height_ratios=[1.25, 1], hspace=0.12)
        ax_s = fig.add_subplot(cell[0])
        schematic(ax_s, ir)
        states = int(sum(np.prod([sizes(ir)[a] for a in f["axes"]]) for f in ir["fields"]) + len(ir["compartments"]))
        ax_s.set_title(f"{key}  {title}\n{ref}; {states:,} states", fontsize=6.5, loc="left", pad=2)
        ax_t = fig.add_subplot(cell[1])
        thumbnail(ax_t, ir, run, view)
        panels += [ax_s, ax_t]
    ps.check(fig, panels)
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figS1_gallery.{ext}")
    print(f"wrote {OUT / 'figS1_gallery.png'} (TenQSP {version})")


if __name__ == "__main__":
    main()
