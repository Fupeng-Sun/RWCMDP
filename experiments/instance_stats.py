"""Descriptive numbers of the fleet maintenance instance quoted in Section 5 (runs in about a minute).

1. Budget tightness: in how many of the T = 30 periods the optimal solution of FLP(p0) exhausts the repair capacity
   (beta = 0.25 and 0.5), and the mean budget multiplier.
2. Data scarcity: for K trajectories per machine (histories 0-19), the share of the state-action pairs with more than
   one reachable state (`do nothing` at ages 2-7) that are never observed, and the share of the pairs whose MLE assigns
   zero probability to a shock failure, by age.
Usage:  python experiments/instance_stats.py [--K 5 10 20] [--seeds 0 1 ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._common import parser  # noqa: E402

import numpy as np  # noqa: E402

from rwcmdp import instance, lp  # noqa: E402
from rwcmdp.ambiguity import ambiguous_pairs  # noqa: E402
from rwcmdp.data import history  # noqa: E402


def budget_tightness(beta):
    P0, R, mask, q = instance.fleet()
    _, x, mu = lp.solve_fluid_lp(P0, R, q, beta, instance.HORIZON)
    repairs = x[..., instance.REPAIR].sum(axis=(1, 2))
    binding = repairs >= beta * instance.N_MACHINES - 1e-6
    return int(binding.sum()), float(mu.mean())


def data_scarcity(K, seeds):
    unobserved, zero_shock, by_age = [], [], []
    for s in seeds:
        h = history(K, s)
        amb = ambiguous_pairs(h.mask)                       # (I, S, A): do nothing at ages 2-7
        n = h.N.sum(-1)
        unobserved.append((n[amb] == 0).mean())
        rows = h.N[:, 1:instance.N_AGES, instance.DO_NOTHING, :]       # ages 2-7
        visited = rows.sum(-1) > 0
        no_shock = rows[..., instance.FAILED] == 0
        zero_shock.append((visited & no_shock).mean())
        by_age.append((visited & no_shock).mean(axis=0))
    return float(np.mean(unobserved)), float(np.mean(zero_shock)), np.mean(by_age, axis=0)


if __name__ == "__main__":
    p = parser(__doc__, outputs=False)
    p.add_argument("--K", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    a = p.parse_args()
    for beta in (0.25, 0.5):
        n_bind, mu_mean = budget_tightness(beta)
        print(f"beta = {beta}: FLP(p0) exhausts the repair capacity in {n_bind} of {instance.HORIZON} periods "
              f"(mean budget multiplier {mu_mean:.3f})")
    for K in a.K:
        u, z, ages = data_scarcity(K, a.seeds)
        print(f"K = {K:2d} ({len(a.seeds)} histories): never observed {100 * u:.1f}% of the ambiguous pairs; "
              f"MLE shock probability 0 at {100 * z:.1f}% of them (observed, no shock seen); by age 2-7: "
              + " ".join(f"{100 * v:.0f}%" for v in ages))
