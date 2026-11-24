# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The spheroid virtual population of Table S3, one member at a time (Supplement S9).

Each of the 64 members of tables/tableS3_batches.py is solved alone with the JAX backend and the
same settings; the members that fail are solved again with the scipy backend (which retries a
failing segment with Radau IIA). Writes results/tableS3/spheroid_members.csv: for each member
that failed alone, its kill rate k1 and whether scipy solved it, with the TenQSP version.

    python checks/spheroid_members.py      # about 15 min
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tables"))
sys.path.insert(0, str(ROOT / "figures"))
import tableS3_batches as tb
import tenqsp as tq

if __name__ == "__main__":
    m, sc, wrt, outputs, opts, members, _ = tb.spheroid()
    g = m.simulator(outputs=outputs, backend="jax", **opts).function(sc, wrt=wrt, derivatives=False)
    x0 = np.asarray(g.x0)
    X = x0 * np.exp(0.3 * np.random.default_rng(0).standard_normal((members, x0.size)))  # as the batch
    one = jax.jit(lambda x: g(x)[outputs[0]])
    bad = [i for i in range(members) if np.isnan(np.asarray(one(jnp.asarray(X[i])))).any()]
    ss = m.simulator(outputs=outputs, backend="scipy", **opts)
    k = wrt.index("k1")
    rows = ["member,k1,failed_alone_jax,scipy_solved,tenqsp"]
    for i in range(members):
        solved = ""
        if i in bad:
            p = dict(sc.params)
            p.update(dict(zip(wrt, X[i], strict=True)))
            r = np.asarray(ss(tq.Scenario(params=p, regimen=sc.regimen, times=sc.times))[outputs[0]])
            solved = int(np.isfinite(r).all())
        rows.append(f"{i},{X[i][k]:.4f},{int(i in bad)},{solved},{tq.__version__}")
    (ROOT / "results" / "tableS3" / "spheroid_members.csv").write_text("\n".join(rows) + "\n")
    print(f"failed alone: {bad}; typical k1 {x0[k]:.2f}")
