"""Section 4.4, last paragraph: with asymmetric optimal solutions of FLP(p~), an unfavorable order of the reverted
machines among those with equal index gains leads to a capacity shortfall that grows linearly in the number of machines.

Stylized model of the marginal class of the rare-failure instance of Section 4.4: n working machines in K = 6 groups
(random sizes), capacity B = 0.4 n per period, horizon T = 8. The plan is a feasible fluid solution in which group k
requests maintenance with probability Z[t, k] (planned failure probability 0.15 per unmaintained period, so the plan
uses the whole capacity in every period). Plans are vertex-like ('greedy': groups filled one after another in random
order) or interior ('frac': capacity spread over all groups). Under the true failure probability 0.02, machines draw
their requests from Z; excess requests are reverted with sticky restoration, either uniformly at random ('random'),
reverting the machines with the largest planned future maintenance first ('maxfut'), or the smallest first
('minfut'). The script prints the expected capacity shortfall sum_t E(B - granted_t)_+ for n in {200, 800, 3200},
averaged over 5 plans (100 runs each), and its ratio to sqrt(n): a ratio that grows with n indicates linear growth.
Runs in a few seconds.  Usage:  python experiments/sec44_tie_order.py
"""
import numpy as np

T, PLANNED_FAILURE, TRUE_FAILURE = 8, 0.15, 0.02
N_GROUPS, CAPACITY_SHARE, N_PLANS, N_RUNS = 6, 0.4, 5, 100


def random_plan(K, sizes, B, nk, T, rng, mode):
    """Request probabilities Z (T, K) of a fluid plan that uses the whole capacity B in periods 1..T-1."""
    w = np.ones(K)          # planned share of working machines of every group
    Z = np.zeros((T, K))
    for t in range(T - 1):
        cap = B
        mass = sizes * w
        if mode == "greedy":
            for k in rng.permutation(K):
                z = min(cap, mass[k])
                Z[t, k] = z / mass[k]
                cap -= z
        else:
            f = rng.random(K)
            f = f / f.sum() * B
            f = np.minimum(f, mass)
            while B - f.sum() > 1e-9:
                room = mass - f
                add = np.minimum(room, (B - f.sum()) * room / room.sum())
                f += add
            Z[t] = f / mass
        assert abs((Z[t] * mass).sum() - B) < 1e-6
        w = w * (Z[t] + (1 - Z[t]) * (1 - nk))
    return Z


def shortfall(n, B, nu, T, Z, sizes, rule, runs, rng):
    """Mean capacity shortfall sum_t (B - granted_t)_+ under sticky restoration with the given revert rule."""
    group = np.repeat(np.arange(len(sizes)), sizes)
    future = np.cumsum(Z[::-1], 0)[::-1]   # planned maintenance from period t on
    working = np.ones((runs, n), bool)
    frozen = np.zeros((runs, n), bool)
    short = np.zeros(runs)
    for t in range(T - 1):
        req = working & ~frozen & (rng.random((runs, n)) < Z[t, group])
        short += np.maximum(B - req.sum(1), 0)
        if rule == "random":
            priority = rng.random((runs, n))
        else:
            v = future[t + 1, group] if t + 1 < T else np.zeros(n)
            priority = (-v if rule == "maxfut" else v) + 1e-6 * rng.random((runs, n))
        priority = np.where(req, priority, -np.inf)
        rank = np.argsort(np.argsort(-priority, 1), 1)
        granted = req & (rank < B)
        frozen |= req & ~granted
        working = working & (granted | (rng.random((runs, n)) >= nu))
    return short.mean()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for mode in ("greedy", "frac"):
        for rule in ("random", "maxfut", "minfut"):
            res = []
            for n in (200, 800, 3200):
                B = int(CAPACITY_SHARE * n)
                acc = []
                for rep in range(N_PLANS):
                    sizes = rng.multinomial(n - N_GROUPS, np.ones(N_GROUPS) / N_GROUPS) + 1
                    Z = random_plan(N_GROUPS, sizes, B, PLANNED_FAILURE, T, np.random.default_rng(rep), mode)
                    acc.append(shortfall(n, B, TRUE_FAILURE, T, Z, sizes, rule, N_RUNS, rng))
                res.append(f"n={n}: {np.mean(acc):7.2f} (/sqrt {np.mean(acc) / np.sqrt(n):.3f}, "
                           f"max/sqrt {np.max(acc) / np.sqrt(n):.3f})")
            print(mode, rule, " | ".join(res), flush=True)
