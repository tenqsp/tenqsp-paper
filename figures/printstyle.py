# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Shared print style of the paper's figures: drawn at the journal's full width (7 in), text at
6-7 pt, saved at 300 dpi; `check` stops a script if a panel's text leaves the figure or runs
into a neighboring panel, measured at the save resolution."""

import matplotlib.pyplot as plt

WIDTH = 7.0  # inches, full page width
DPI = 300
SIZE = 6.5  # points


def setup():
    plt.rcParams.update({"font.size": SIZE, "axes.titlesize": 7.5, "axes.titleweight": "bold",
                         "axes.titlelocation": "left", "axes.labelsize": SIZE + 0.5,
                         "legend.fontsize": SIZE - 0.5, "legend.frameon": False, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
                         "ytick.minor.width": 0.4, "lines.linewidth": 1.1, "lines.markersize": 3.5,
                         "figure.dpi": DPI, "savefig.dpi": DPI})


def check(fig, panels):
    """Stop if the drawn extent of any panel (axes with its labels, ticks, title and legend)
    leaves the figure or overlaps another panel's."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [((ax.get_title(loc="left")[:20] if hasattr(ax, "get_xlim") else "legend") or str(i),
              ax.get_tightbbox(r)) for i, ax in enumerate(panels)]
    page = fig.bbox
    for name, b in boxes:
        if b.x0 < page.x0 - 1 or b.y0 < page.y0 - 1 or b.x1 > page.x1 + 1 or b.y1 > page.y1 + 1:
            raise SystemExit(f"panel {name!r} runs off the figure")
    for i, (n1, b1) in enumerate(boxes):
        for n2, b2 in boxes[i + 1:]:
            if b1.x0 < b2.x1 - 1 and b2.x0 < b1.x1 - 1 and b1.y0 < b2.y1 - 1 and b2.y0 < b1.y1 - 1:
                raise SystemExit(f"panels {n1!r} and {n2!r} overlap")
