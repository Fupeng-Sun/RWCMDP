"""Runtimes quoted in Section 5 for one history with K = 5 (seed 0, beta = 0.5, c = 0.5): the fluid LP of the MLE
policy, the CDR LP, 150 iterations of Algorithm 1 and one simulation of 1,000 runs of the fleet.
Usage:  python experiments/timing.py [--repeat 3]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._common import parser  # noqa: E402

import platform  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from rwcmdp import evaluation as ev  # noqa: E402
from rwcmdp import lp  # noqa: E402
from rwcmdp.ambiguity import lower_envelope, radius  # noqa: E402
from rwcmdp.data import history  # noqa: E402
from rwcmdp.instance import HORIZON  # noqa: E402
from rwcmdp.policies import fluid_policy  # noqa: E402
from rwcmdp.relaxation import subgradient_method  # noqa: E402


def timed(fn, repeat):
    times, out = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), out


if __name__ == "__main__":
    p = parser(__doc__, outputs=False)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--c", type=float, default=0.5)
    p.add_argument("--repeat", type=int, default=3)
    a = p.parse_args()
    h = history(a.K, a.seed)
    rho = radius(a.c, a.K, h.L)
    print(f"{platform.platform()}, {platform.processor() or platform.machine()}, Python {platform.python_version()}")
    t_flp, (_, x, mu) = timed(lambda: lp.solve_fluid_lp(h.P_hat, h.R, h.q, a.beta, HORIZON), a.repeat)
    print(f"fluid LP FLP(p_hat):                {t_flp:6.2f} s")
    P_low = lower_envelope(h.P_hat, rho, h.mask)
    t_cdr, _ = timed(lambda: lp.solve_cdr_lp(P_low, h.R, h.q, a.beta, HORIZON), a.repeat)
    print(f"CDR LP:                             {t_cdr:6.2f} s")
    t_sub, _ = timed(lambda: subgradient_method(h.P_hat, rho, mu, h.R, a.beta, h.mask, h.q, HORIZON), a.repeat)
    print(f"Algorithm 1, 150 iterations:        {t_sub:6.2f} s")
    prob = fluid_policy(x)
    t_sim, _ = timed(lambda: ev.reward(h, a.beta, prob), a.repeat)
    print(f"simulation, {ev.SIM_RUNS} runs of the fleet:   {t_sim:6.2f} s")
