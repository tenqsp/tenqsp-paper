# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""Table 3, virtual populations: many parameter sets of one model run in one call
(jax.vmap of sim.function(..., derivatives=False)), against one call per set, on one CPU
core, on 64 CPU cores (jax.shard_map) and on a GPU. Writes results/table3/batch_*.json.

    python tables/table3_batches.py                 # every CPU case, one process each
    python tables/table3_batches.py gpu             # every GPU case (an environment with CUDA JAX)
    python tables/table3_batches.py friberg vmap    # one case

Every member draws its parameters from a log-normal distribution (SD 0.3 on the log scale)
around the model's values. "loop": one compiled call per member (the mean over a few);
"vmap": all members in one call on one core; "cores": the same call split over 64 cores;
"gpu": on the GPU. Times: the first batch (which compiles) and a second one, wall-clock.
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "table3"
sys.path.insert(0, str(ROOT / "figures"))
CORES = 64


def friberg():
    import tenqsp as tq

    m = tq.Model.from_text((ROOT / "models" / "friberg_maturation.tq").read_text())
    sc = tq.Scenario(regimen=tq.bolus("Ac", 90, times=[0]), times=24.0 * np.array([1, 3, 7, 9, 11, 14, 18, 21]))
    return m, sc, ["MTT", "gamma", "circ0", "slope"], ["ANC"], {"rtol": 1e-6, "atol": 1e-9}, 1024, 16


def pbpk():
    import tenqsp as tq

    m = tq.Model.from_text((ROOT / "models" / "pbpk_wholebody.tq").read_text())
    sc = tq.Scenario(regimen=tq.bolus("Cven", 100.0, times=[0]), times=np.linspace(1, 96, 24))
    return m, sc, ["PS1", "PS2", "CLint", "fu"], ["Cven"], {"rtol": 1e-6, "atol": 1e-9}, 1024, 16


def spheroid():
    import spheroid as sp
    import tenqsp as tq

    sp.GRIDS["table3"] = (10, 25)
    m = sp.model("spheroid_pk", "table3")
    sc = tq.Scenario(params={"isat": 0.25}, regimen=sp.regimen("21"), times=np.linspace(50, sp.T, 20))
    return m, sc, ["s1", "k1", "epsd", "isat"], ["total"], sp.SOLVER, 64, 4


MODELS = {"friberg": ("Friberg myelosuppression", friberg), "pbpk": ("whole-body PBPK", pbpk),
          "spheroid": ("tumour spheroid, 10 x 25 cells", spheroid)}
CPU_MODES = ("loop", "vmap", "cores")


def run(name, mode):
    if mode == "cores":
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={CORES}"
    import jax
    import jax.numpy as jnp
    import tenqsp as tq

    label, build = MODELS[name]
    m, sc, wrt, outputs, opts, members, loop_over = build()
    sim = m.simulator(outputs=outputs, backend="jax", **opts)
    g = sim.function(sc, wrt=wrt, derivatives=False)
    x0 = np.asarray(g.x0)
    rng = np.random.default_rng(0)
    X = jnp.asarray(x0 * np.exp(0.3 * rng.standard_normal((members, x0.size))))

    def h(x):
        return g(x)[outputs[0]]

    out = {"model": label, "key": name, "mode": mode, "device": jax.devices()[0].platform,
           "states": int(m.compile().n_states), "members": members}
    if mode == "loop":
        one = jax.jit(h)
        t0 = time.perf_counter()
        one(X[0]).block_until_ready()
        out["first_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        for x in X[1:loop_over + 1]:
            one(x).block_until_ready()
        out["per_member_s"] = (time.perf_counter() - t0) / loop_over
        out["members_timed"] = loop_over
        r = None
    else:
        if mode == "cores":
            from jax.sharding import Mesh
            from jax.sharding import PartitionSpec as P

            mesh = Mesh(np.array(jax.devices()), ("p",))
            batch = jax.jit(jax.shard_map(jax.vmap(h), mesh=mesh, in_specs=P("p"), out_specs=P("p"),
                                          check_vma=False))
            out["cores"] = CORES
        else:
            batch = jax.jit(jax.vmap(h))
        t0 = time.perf_counter()
        batch(X).block_until_ready()
        out["first_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        r = batch(X).block_until_ready()
        out["batch_s"] = time.perf_counter() - t0
        out["per_member_s"] = out["batch_s"] / members
        out["failed"] = int(np.isnan(np.asarray(r)).reshape(members, -1).any(axis=1).sum())
    out["peak_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    if out["device"] == "gpu":
        out["peak_gpu_gb"] = (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 1e9
        out["gpu"] = jax.devices()[0].device_kind
    out["versions"] = {"tenqsp": tq.__version__, "jax": jax.__version__, "python": platform.python_version()}
    out["load_at_start"] = os.getloadavg()[0]
    return out


def main():
    args = sys.argv[1:]
    if args[:1] == ["--child"]:  # one case, in its own process (the device count is fixed at import)
        print(json.dumps(run(*args[1:])))
        return
    gpu = args == ["gpu"]
    if len(args) == 2:
        cases = [tuple(args)]
    else:
        cases = [(name, mode) for name in MODELS for mode in (("vmap",) if gpu else CPU_MODES)]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, mode in cases:
        p = subprocess.run([sys.executable, __file__, "--child", name, mode], capture_output=True, text=True,
                           check=False)
        if p.returncode != 0:
            print(f"{name} {mode} failed:\n{p.stderr[-2000:]}", flush=True)
            continue
        row = json.loads(p.stdout.strip().splitlines()[-1])
        tag = "gpu" if row["device"] == "gpu" else mode
        (OUT / f"batch_{name}_{tag}.json").write_text(json.dumps(row, indent=1))
        print(f"{row['model']} ({row['states']} states), {tag}: {row['per_member_s']:.4g} s per member "
              f"(first call {row['first_s']:.3g} s), {row['peak_gb']:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
