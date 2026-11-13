"""Decomposition of the optimality gaps in Section 5 (the identity of Section 4.4).

For mu* optimal in the dual of FLP(p0) and Delta_ti(s, a) = W_ti(s) - Q_ti(s, a) under (p0, mu*), every policy satisfies
    L*(p0) - V(pi; p0) = sum_t mu*_t (b_t - E[C_t]) + sum_{t,i} E[Delta_ti(S_ti, A_ti)],
where C_t is the number of repairs in period t. The script simulates the MLE, subgradient, CDR and true-kernel
policies under p0 (period-by-period restoration, the evaluation stream of Section 5) and records, per policy, the
mean total reward, the priced unused capacity sum_t mu*_t (b_t - E[C_t]) and the action regret sum E[Delta], all
over the 1,000 evaluation runs; the identity holds up to the simulation error.
Default histories: K = 5, beta in {0.25, 0.5}, seeds 0-2, radius multiplier c = 2 for both robust policies (as in
Table 2). Writes text_decomposition.pkl to --out-dir; about 1 min per history with one worker.
Quick check:  python experiments/text_decomposition.py --beta 0.25 --seeds 0 --workers 1
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._common import parser, run_tasks  # noqa: E402

from rwcmdp import evaluation as ev  # noqa: E402
from rwcmdp import lp  # noqa: E402
from rwcmdp.ambiguity import radius  # noqa: E402
from rwcmdp.data import history, sample_next_states  # noqa: E402
from rwcmdp.instance import HORIZON  # noqa: E402
from rwcmdp.policies import TIE_TOL, cdr_policy, constant_sequence, fluid_policy, index_policy, subgradient_policy  # noqa: E402


def simulate_decomposition(seed, kind, pol, P_true, R, q, cap, n_runs, mu, gain):
    """Period-by-period restoration as in rwcmdp.simulation; kind 'prob' (repair probabilities) or 'index' (IndexPolicy).

    Returns (mean total reward, sum_t mu_t (cap - E[C_t]), sum_{t,i} E[Delta]), with Delta from the index gains
    gain = Q(s, repair) - Q(s, do nothing) under (p0, mu*): Delta(s, repair) = (-gain)_+, Delta(s, do nothing) = gain_+.
    """
    rng = np.random.default_rng(seed)
    T, I = HORIZON, P_true.shape[0]
    machine = np.arange(I)
    state = sample_next_states(np.broadcast_to(q, (n_runs, I, q.shape[-1])), rng.random((n_runs, I)))
    total, unused, regret = np.zeros(n_runs), 0.0, 0.0
    for t in range(T):
        if kind == "prob":
            wants = rng.random((n_runs, I)) < pol[t, machine, state]
            priority = np.where(wants, rng.random((n_runs, I)), -1.0)
        else:
            idx = pol.index[t, machine, state]
            u = rng.random((n_runs, I))
            wants = (idx > TIE_TOL) | ((np.abs(idx) <= TIE_TOL) & (u < pol.tie_prob[t, machine, state]))
            priority = np.where(wants, np.where(idx > TIE_TOL, idx, 0.0) + 1e-9 * rng.random((n_runs, I)), -np.inf)
        rank = np.argsort(np.argsort(-priority, axis=1), axis=1)
        action = (wants & (rank < cap)).astype(int)
        total += R[state, action].sum(axis=1)
        unused += mu[t] * (cap - action.sum(axis=1).mean())
        g = gain[t, machine, state]
        regret += np.where(action == 1, np.maximum(-g, 0.0), np.maximum(g, 0.0)).sum(axis=1).mean()
        state = sample_next_states(P_true[machine, state, action], rng.random((n_runs, I)))
    return float(total.mean()), float(unused), float(regret)


def task(K, seed, beta, c):
    h = history(K, seed)
    cap, stream, runs = ev.capacity(beta, h), ev.eval_stream(seed), ev.SIM_RUNS
    L_star, x0, mu_star = lp.solve_fluid_lp(h.P0, h.R, h.q, beta, HORIZON)
    gain = index_policy(constant_sequence(h.P0, HORIZON), h.R, h.q, beta, HORIZON).index   # Q-values under (p0, mu*)
    _, x_mle, mu_mle = lp.solve_fluid_lp(h.P_hat, h.R, h.q, beta, HORIZON)
    rho = radius(c, K, h.L)
    sub = subgradient_policy(h.P_hat, rho, h.R, h.q, beta, h.mask, HORIZON, mu_mle)
    _, p_cdr = cdr_policy(h.P_hat, rho, h.R, h.q, beta, h.mask, HORIZON)
    out = dict(K=K, seed=seed, b=beta, c=c, L_star=float(L_star), mu_star=mu_star)
    for name, kind, pol in (("TRUE", "prob", fluid_policy(x0)), ("MLE", "prob", fluid_policy(x_mle)),
                            ("Sub", "index", sub), ("CDR", "prob", p_cdr)):
        reward, unused, regret = simulate_decomposition(stream, kind, pol, h.P0, h.R, h.q, cap, runs, mu_star, gain)
        out[name] = dict(reward=reward, unused=unused, regret=regret)
    return out


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--K", type=int, nargs="+", default=[5])
    p.add_argument("--beta", type=float, nargs="+", default=[0.25, 0.5])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--c", type=float, default=2.0, help="radius multiplier of both robust policies (default 2)")
    a = p.parse_args()
    tasks = [(K, s, b, a.c) for K in a.K for b in a.beta for s in a.seeds]
    run_tasks(task, tasks, a.workers, os.path.join(a.out_dir, "text_decomposition.pkl"), lambda t: str(t[:3]))
