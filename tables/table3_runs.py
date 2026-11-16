# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Table 3, what a run costs: the case-study models from 3 to about 17,000 states, solved by
the scipy backend and by the JAX backend, and the gradient of a least-squares objective
with respect to the model's parameters. Writes results/table3/<case>.json.

    python tables/table3_runs.py           # every case, one process each, one after another
    python tables/table3_runs.py 3         # one case (index into CASES)

Per case: "build" turns the model into its program (once per model); "scipy" and "JAX" are
one solve at the scenario; "f(x)" one call of the model as a JAX function of the parameters
(sim.function); "gradient" jax.grad of the sum of squares of its outputs, the objective of
an estimator. Each is timed at a first call (for JAX it includes compiling) and then as the
best of 3 calls (of 1 above 2,000 states). "gap": the largest difference between the two
backends' outputs, relative to each output's largest value. Run alone on an
idle machine: the times are wall-clock seconds.
"""

import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results" / "table3"
sys.path.insert(0, str(ROOT / "figures"))

PK = """model pk
param ka = 1.2, CL = 4, V = 30, Q = 8, V2 = 60
compartment depot, centr, periph
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - CL/V*centr - Q/V*centr + Q/V2*periph
d_t(periph) = Q/V*centr - Q/V2*periph
observable conc = centr/V
"""
DEFAULT = {"rtol": 1e-6, "atol": 1e-9}


def pk():
    import tenqsp as tq

    reg = tq.bolus("depot", 100, times=[0, 12])
    return tq.Model.from_text(PK), reg, np.linspace(1, 24, 24), ["ka", "CL", "V", "Q", "V2"], ["conc"], DEFAULT


def transit():
    import tenqsp as tq
    import transit_fit as tr

    return tr.transit_model(25), tq.Regimen(), tr.TIMES, list(tr.TRUE), ["conc"], DEFAULT


def dispersion():
    import tenqsp as tq
    import transit_fit as tr

    return (tr.transit_model(200, True), tq.Regimen(), tr.TIMES, [*tr.TRUE, "disp"], ["conc"], DEFAULT)


def friberg():
    import tenqsp as tq

    m = tq.Model.from_text((ROOT / "models" / "friberg_maturation.tq").read_text())
    reg = tq.bolus("Ac", 90, times=[0])
    return m, reg, 24.0 * np.array([1, 3, 7, 9, 11, 14, 18, 21]), ["MTT", "gamma", "circ0", "slope"], ["ANC"], DEFAULT


def eclipse():
    import eclipse as ec
    import tenqsp as tq

    return (ec.eclipse_model(40), tq.Regimen(), np.linspace(0.5, 10, 20), ["beta", "p", "c", "delta", "V0"],
            ["log10V"], DEFAULT)


def pbpk():
    import tenqsp as tq

    m = tq.Model.from_text((ROOT / "models" / "pbpk_wholebody.tq").read_text())
    reg = tq.bolus("Cven", 100.0, times=[0])
    return m, reg, np.linspace(1, 96, 24), ["PS1", "PS2", "CLint", "fu"], ["Cven", "in_tissues"], DEFAULT


def spheroid(nr, nx):
    def build():
        import spheroid as sp

        sp.GRIDS["table3"] = (nr, nx)
        m = sp.model("spheroid_pk", "table3")
        return (m, sp.regimen("21"), np.linspace(50, sp.T, 20), ["s1", "k1", "epsd", "isat"], ["total", "mean_x"],
                {**sp.SOLVER, "params": {"isat": 0.25}})

    return build


# (label, figure, build)
CASES = [
    ("two-compartment oral PK, 2 doses", "-", pk),
    ("Friberg myelosuppression, 3 maturation cells", "5", friberg),
    ("transit absorption, chain of 25", "5", transit),
    ("whole-body PBPK, 9 organs x 3 spaces", "6", pbpk),
    ("influenza, eclipse on 40 cells", "2", eclipse),
    ("transit with dispersion, 200 cells (sharp)", "5", dispersion),
    ("tumour spheroid, 10 x 25 cells", "7", spheroid(10, 25)),
    ("tumour spheroid, 20 x 50 cells", "7", spheroid(20, 50)),
    ("tumour spheroid, 40 x 100 cells", "7", spheroid(40, 100)),
    ("tumour spheroid, 80 x 200 cells", "7", spheroid(80, 200)),
]


def timed(fn, k=3):
    """The first call, then the best of k."""
    import jax

    t0 = time.perf_counter()
    jax.block_until_ready(fn())
    first = time.perf_counter() - t0
    walls = []
    for _ in range(k):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        walls.append(time.perf_counter() - t0)
    return first, min(walls)


def run(i):
    import jax
    import jax.numpy as jnp
    import tenqsp as tq

    label, fig, build = CASES[i]
    m, reg, times, wrt, outputs, opts = build()
    opts = dict(opts)
    params = opts.pop("params", None)
    sc = tq.Scenario(params=params or {}, regimen=reg, times=np.asarray(times, float))
    t0 = time.perf_counter()
    cm = m.compile()
    build_s = time.perf_counter() - t0

    sim_s = m.simulator(outputs=outputs, backend="scipy", **opts)
    sim_j = m.simulator(outputs=outputs, backend="jax", **opts)
    k = 3 if cm.n_states < 2000 else 1  # repeats after the first call (large models: one)
    scipy_first, scipy_s = timed(lambda: sim_s(sc)[outputs[0]], k)
    jax_first, jax_s = timed(lambda: sim_j(sc)[outputs[0]], k)
    a, b = sim_s(sc), sim_j(sc)
    gap = max(float(np.max(np.abs(np.asarray(a[k]) - np.asarray(b[k]))) / np.max(np.abs(np.asarray(a[k]))))
              for k in outputs)

    f = sim_j.function(sc, wrt=wrt)
    x0 = jnp.asarray(f.x0)
    f_first, f_s = timed(lambda: f(x0)[outputs[0]], k)

    def loss(x):
        return sum(jnp.sum(v**2) for v in f(x).values())

    grad = jax.jit(jax.grad(loss))
    grad_first, grad_s = timed(lambda: grad(x0), k)
    g = np.asarray(grad(x0))
    return {
        "label": label, "figure": fig, "repeats": k, "states": int(cm.n_states), "params": len(wrt),
        "derivatives": f.derivatives, "build_s": build_s,
        "scipy_first_s": scipy_first, "scipy_s": scipy_s,
        "jax_first_s": jax_first, "jax_s": jax_s, "gap": gap,
        "f_first_s": f_first, "f_s": f_s, "grad_first_s": grad_first, "grad_s": grad_s,
        "grad_finite": bool(np.all(np.isfinite(g))),
        "peak_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        "versions": {"tenqsp": tq.__version__, "jax": jax.__version__, "python": platform.python_version()},
        "load_at_start": os.getloadavg()[0],
    }


def main():
    if len(sys.argv) > 1:
        print(json.dumps(run(int(sys.argv[1]))))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for i in range(len(CASES)):
        p = subprocess.run([sys.executable, __file__, str(i)], capture_output=True, text=True, check=False)
        if p.returncode != 0:
            print(f"case {i} failed:\n{p.stderr[-2000:]}", flush=True)
            continue
        row = json.loads(p.stdout.strip().splitlines()[-1])
        (OUT / f"case{i:02d}.json").write_text(json.dumps(row, indent=1))
        print(f"{row['label']}: {row['states']:,} states, scipy {row['scipy_s']:.3g} s, JAX {row['jax_s']:.3g} s "
              f"(first {row['jax_first_s']:.3g}), gradient {row['grad_s']:.3g} s (first {row['grad_first_s']:.3g}), "
              f"{row['peak_gb']:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
