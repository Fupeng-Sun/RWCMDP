"""Control policies and the restoration footnote of Section 5.

kind 'fig4'  (every history of Figure 4, beta > 0): policy ratios of
    'MLE-index'       the index policy of Definition 5 for the constant kernel p_hat (= subgradient policy at c = 0),
    'TRUE-index'      the same index policy for the true kernel p0,
    'random-working'  the random policy that requests repairs only for working machines.
kind 'sticky' (K = 5, beta in {0.25, 0.5}, histories 0-4): MLE, CDR and subgradient policies with the validated
    multipliers of Figure 4, under period-by-period restoration ('period') and sticky restoration ('sticky', reverted
    machines keep doing nothing; LP-based policies revert at random, the subgradient policy by smallest index gain).
The validated multipliers are read from the Figure 4 results in --fig4-dir (default: the stored results/).
Writes text_controls.pkl to --out-dir. Full run: 130 tasks, about 10 min with 6 workers.
Quick check:  python experiments/text_controls.py --K 5 --beta 0.25 --seeds 0
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._common import ROOT, parser, run_tasks  # noqa: E402
from experiments.fig4_cross_validation import default_histories  # noqa: E402

from rwcmdp import evaluation as ev  # noqa: E402
from rwcmdp.ambiguity import radius  # noqa: E402
from rwcmdp.data import history  # noqa: E402
from rwcmdp.instance import HORIZON  # noqa: E402
from rwcmdp.policies import (cdr_policy, constant_sequence, fluid_lp_policy, index_policy,  # noqa: E402
                             random_working_policy, subgradient_policy)
from rwcmdp.results_io import load_fig4  # noqa: E402
from rwcmdp.simulation import simulate, simulate_index, simulate_index_sticky, simulate_sticky  # noqa: E402


def fig4_task(K, seed, beta):
    h = history(K, seed)
    v_true = ev.true_kernel_reward(h, beta)
    out = dict(kind="fig4", K=K, seed=seed, b=beta)
    out["MLE-index"] = ev.reward_index(h, beta, index_policy(constant_sequence(h.P_hat, HORIZON), h.R, h.q, beta, HORIZON)) / v_true
    out["TRUE-index"] = ev.reward_index(h, beta, index_policy(constant_sequence(h.P0, HORIZON), h.R, h.q, beta, HORIZON)) / v_true
    out["random-working"] = ev.reward(h, beta, random_working_policy(HORIZON, h.n_machines, h.P0.shape[1])) / v_true
    return out


def sticky_task(K, seed, beta, c_sub, c_cdr):
    h = history(K, seed)
    cap, sim, runs = ev.capacity(beta, h), ev.eval_stream(seed), ev.SIM_RUNS
    v_true = ev.true_kernel_reward(h, beta)
    out = dict(kind="sticky", K=K, seed=seed, b=beta, c_sub=c_sub, c_cdr=c_cdr)
    _, p_mle, mu_mle = fluid_lp_policy(h.P_hat, h.R, h.q, beta, HORIZON)
    _, p_cdr = cdr_policy(h.P_hat, radius(c_cdr, K, h.L), h.R, h.q, beta, h.mask, HORIZON)
    sub = subgradient_policy(h.P_hat, radius(c_sub, K, h.L), h.R, h.q, beta, h.mask, HORIZON, mu_mle)
    for name, prob in (("MLE", p_mle), ("CDR", p_cdr)):
        out[name] = dict(period=simulate(sim, prob, h.P0, h.R, h.q, cap, runs) / v_true,
                         sticky=simulate_sticky(sim, prob, h.P0, h.R, h.q, cap, runs)[0] / v_true)
    out["Sub"] = dict(period=simulate_index(sim, sub, h.P0, h.R, h.q, cap, runs) / v_true,
                      sticky=simulate_index_sticky(sim, sub, h.P0, h.R, h.q, cap, runs) / v_true)
    return out


def run(kind, *args):
    return {"fig4": fig4_task, "sticky": sticky_task}[kind](*args)


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--fig4-dir", default=os.path.join(ROOT, "results"), help="Figure 4 results with the validated c")
    p.add_argument("--K", type=int, nargs="+", default=None)
    p.add_argument("--beta", type=float, nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--kind", choices=["fig4", "sticky", "both"], default="both")
    a = p.parse_args()
    keep = lambda K, s, b: ((a.K is None or K in a.K) and (a.seeds is None or s in a.seeds)  # noqa: E731
                            and (a.beta is None or any(abs(b - x) < 1e-9 for x in a.beta)))
    tasks = []
    if a.kind in ("fig4", "both"):
        tasks += [("fig4", K, s, b) for K, s, b in default_histories() if b > 0 and keep(K, s, b)]
    if a.kind in ("sticky", "both"):
        cv = {(r["K"], r["seed"], round(r["b"], 4)): r for r in load_fig4(a.fig4_dir)}
        tasks += [("sticky", 5, s, b, cv[(5, s, b)]["c_sub"], cv[(5, s, b)]["c_cdr"])
                  for b in (0.25, 0.5) for s in range(5) if keep(5, s, b)]
    run_tasks(run, tasks, a.workers, os.path.join(a.out_dir, "text_controls.pkl"), lambda t: str(t[:4]))
