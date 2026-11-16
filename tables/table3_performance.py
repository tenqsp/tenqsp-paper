# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Table 3 and Figure S-perf, from results/table3 (run table3_runs.py and table3_batches.py
first; the GPU rows need an environment with CUDA JAX).

    python tables/table3_performance.py    # writes out/table3.md, out/table3.csv,
                                           # out/table3_batches.md and out/figS_perf.png/.pdf

Table 3: one run of each model on each backend, and the gradient of a least-squares
objective. Batches: many parameter sets of one model, per member.
Figure S-perf: (A) time of one solve and of one gradient against the number of states;
(B) time per member of a virtual population, by how it is run.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "table3"
OUT = ROOT / "out"
MODES = {"loop": "one call per member (1 core)", "vmap": "one batch (1 core)", "cores": "one batch (64 cores)",
         "gpu": "one batch (GPU)"}


def seconds(s):
    return f"{s:.3f}" if s < 1 else f"{s:.2f}" if s < 10 else f"{s:.0f}"


def runs():
    rows = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("case*.json"))]
    df = pd.DataFrame(rows).sort_values("states")
    df["grad_per_jax"] = df.grad_s / df.jax_s
    return df


def batches():
    rows = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("batch_*.json"))]
    df = pd.DataFrame(rows)
    df["how"] = [("gpu" if d == "gpu" else m) for d, m in zip(df.device, df["mode"], strict=True)]
    return df


def table(df):
    head = ("| model | figure | states | scipy (s) | JAX (s) | JAX first call (s) | gradient (s) | "
            "gradient / JAX | gradient first call (s) | memory (GB) | backends differ by |")
    lines = [head, "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in df.itertuples():
        lines.append(f"| {r.label} | {r.figure} | {r.states:,} | {seconds(r.scipy_s)} | {seconds(r.jax_s)} | "
                     f"{seconds(r.jax_first_s)} | {seconds(r.grad_s)} | {r.grad_per_jax:.1f} | "
                     f"{seconds(r.grad_first_s)} | {r.peak_gb:.1f} | {r.gap:.0e} |")
    return "\n".join(lines)


def batch_table(b):
    lines = ["| model | states | members | " + " | ".join(MODES.values()) + " |",
             "|---|---:|---:|" + "---:|" * len(MODES)]
    for key, g in b.groupby("key", sort=False):
        g = g.set_index("how")
        cells = [f"{g.loc[h, 'per_member_s']:.2g}" if h in g.index else "-" for h in MODES]
        lines.append(f"| {g.model.iloc[0]} | {g.states.iloc[0]:,} | {int(g.members.max()):,} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n\nSeconds per member (parameter set)."


def figure(df, b):
    plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7.5, "axes.titleweight": "bold", "axes.labelsize": 7,
                         "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "lines.linewidth": 1.1})
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), dpi=300)  # the journal's full width
    for col, lab, c in (("scipy_s", "solve, scipy", "C0"), ("jax_s", "solve, JAX", "C1"),
                        ("grad_s", "gradient (JAX)", "C3")):
        ax[0].plot(df.states, df[col], "o-", c=c, ms=2.5, label=lab)
    sharp = df.label.str.contains("sharp")
    ax[0].plot(df.states[sharp], df.grad_s[sharp], "o", c="C3", mfc="none", ms=7, mew=0.7,
               label="gradient, sharp scheme")
    ax[0].set(xscale="log", yscale="log", xlabel="States", ylabel="Seconds")
    ax[0].set_title("A  One run of each model", loc="left")
    ax[0].legend(frameon=False)
    keys = list(dict.fromkeys(b.key))
    w = 0.8 / len(MODES)
    for j, (h, lab) in enumerate(MODES.items()):
        vals = [b[(b.key == k) & (b.how == h)].per_member_s.min() if ((b.key == k) & (b.how == h)).any() else np.nan
                for k in keys]
        ax[1].bar(np.arange(len(keys)) + (j - 1.5) * w, vals, w, label=lab, color=f"C{j}")
    names = [f"{b[b.key == k].model.iloc[0].split(',')[0].replace(' ', chr(10), 1)}\n({b[b.key == k].states.iloc[0]:,} states)" for k in keys]
    ax[1].set_xticks(np.arange(len(keys)), names)
    ax[1].set(yscale="log", ylabel="Seconds per member")
    ax[1].set_title("B  Virtual populations", loc="left")
    ax[1].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=1.5)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figS_perf.{ext}", dpi=300)


def main():
    OUT.mkdir(exist_ok=True)
    df = runs()
    (OUT / "table3.md").write_text(table(df) + "\n")
    df.drop(columns=["versions"]).to_csv(OUT / "table3.csv", index=False)
    print(table(df))
    b = batches()
    if len(b):
        (OUT / "table3_batches.md").write_text(batch_table(b) + "\n")
        print()
        print(batch_table(b))
        figure(df, b)
    v = df.versions.iloc[0]
    print(f"\nTenQSP {v['tenqsp']}, JAX {v['jax']}, Python {v['python']}; load at start "
          f"{df.load_at_start.min():.1f}-{df.load_at_start.max():.1f}")


if __name__ == "__main__":
    main()
