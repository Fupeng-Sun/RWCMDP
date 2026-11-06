"""Table 2: planning values, separable worst-case values and realized rewards of the data-driven policies.

For K in {5, 20}, beta in {0.5, 0.25}, c in {0.5, 2} (the table shows c = 2) and histories 0-2: for the MLE,
subgradient and CDR policies, the planning value (optimal value of FLP(p_hat), FLP(p~) or the CDR LP), the separable
worst-case value V_(pi; P) and restoration allowance Gamma(pi; P) of the unrestored policy, and the realized mean
reward under the true kernel (without restoration, with sticky restoration, with period-by-period restoration); all
relative to the true-kernel policy. The fields realized_* simulate every policy, the subgradient policy included, as a
separable policy (its request probabilities) with random reverts; for the subgradient policy, 'realized_index' is the
period-by-period restoration of Definition 5 (reverts by smallest index gain), which Table 2 reports. The CDR LP states the flow constraints as
equalities, as in the definition of the CDR policy. Writes table2_certificates.pkl to --out-dir; records
{K, seed, b, c, v_true, coverage, covered_all, 'MLE' | 'Sub' | 'CDR': {plan, vrob, allowance, cert, realized_fluid,
realized_sticky, frozen_sticky, realized_period, max_overrequest, Cbar}}, plus 'realized_index' for 'Sub'.
With --sub-index-only, only the index-priority realized reward of the subgradient policy is computed and written to
table2_sub_index.pkl (records {K, seed, b, c, v_true, realized_sub_index}); this supplements the stored
table2_certificates.pkl, which predates the field.
Full run: 24 cases, a few minutes with 6 workers. Quick check:  python experiments/table2_certificates.py --K 5 --beta 0.5 --c 2 --seeds 0
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._common import parser, run_tasks  # noqa: E402

import numpy as np  # noqa: E402

from rwcmdp import evaluation as ev  # noqa: E402
from rwcmdp.ambiguity import covers, radius  # noqa: E402
from rwcmdp.certificates import certificate  # noqa: E402
from rwcmdp.data import history  # noqa: E402
from rwcmdp.instance import HORIZON  # noqa: E402
from rwcmdp.policies import cdr_policy, fluid_lp_policy, subgradient_policy  # noqa: E402
from rwcmdp.simulation import simulate, simulate_sticky  # noqa: E402


def task(K, seed, beta, c):
    h = history(K, seed)
    I, cap, sim, runs = h.n_machines, ev.capacity(beta, h), ev.eval_stream(seed), ev.SIM_RUNS
    rho = radius(c, K, h.L)
    v_true = ev.true_kernel_reward(h, beta)
    cover = covers(h.P_hat, h.P0, rho, h.mask)
    out = dict(K=K, seed=seed, b=beta, c=c, v_true=v_true, coverage=float(cover.mean()), covered_all=bool(cover.all()))
    v_mle, p_mle, mu_mle = fluid_lp_policy(h.P_hat, h.R, h.q, beta, HORIZON)
    pols = {"MLE": (v_mle, p_mle)}
    pols["CDR"] = cdr_policy(h.P_hat, rho, h.R, h.q, beta, h.mask, HORIZON, flow_inequality=False)
    sub = subgradient_policy(h.P_hat, rho, h.R, h.q, beta, h.mask, HORIZON, mu_mle)
    pols["Sub"] = (sub.value, sub.request_probability())
    for name, (plan, prob) in pols.items():
        v_rob, gamma, C_bar = certificate(prob, h.P_hat, rho, h.R, h.q, h.mask, cap)
        sticky = simulate_sticky(sim, prob, h.P0, h.R, h.q, cap, runs)
        out[name] = dict(plan=plan / v_true, vrob=v_rob / v_true, allowance=gamma / v_true, cert=(v_rob - gamma) / v_true,
                         realized_fluid=simulate(sim, prob, h.P0, h.R, h.q, I, runs) / v_true,
                         realized_sticky=float(np.mean(sticky[0])) / v_true, frozen_sticky=float(np.mean(sticky[1])),
                         realized_period=simulate(sim, prob, h.P0, h.R, h.q, cap, runs) / v_true,
                         max_overrequest=float(max(C_bar.max() - cap, 0)), Cbar=C_bar)
    out["Sub"]["realized_index"] = ev.reward_index(h, beta, sub) / v_true
    return out


def sub_index_task(K, seed, beta, c):
    """Realized reward of the subgradient policy with index-priority period-by-period restoration (Table 2)."""
    h = history(K, seed)
    v_true = ev.true_kernel_reward(h, beta)
    _, _, mu_mle = fluid_lp_policy(h.P_hat, h.R, h.q, beta, HORIZON)
    sub = subgradient_policy(h.P_hat, radius(c, K, h.L), h.R, h.q, beta, h.mask, HORIZON, mu_mle)
    return dict(K=K, seed=seed, b=beta, c=c, v_true=v_true, realized_sub_index=ev.reward_index(h, beta, sub) / v_true)


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--K", type=int, nargs="+", default=[5, 20])
    p.add_argument("--beta", type=float, nargs="+", default=[0.5, 0.25])
    p.add_argument("--c", type=float, nargs="+", default=[0.5, 2.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--sub-index-only", action="store_true", help="only the index-priority realized reward of Sub")
    a = p.parse_args()
    tasks = [(K, s, b, c) for K in a.K for b in a.beta for c in a.c for s in a.seeds]
    label = lambda t: f"K={t[0]} seed={t[1]} beta={t[2]:g} c={t[3]:g}"  # noqa: E731
    if a.sub_index_only:
        run_tasks(sub_index_task, tasks, a.workers, os.path.join(a.out_dir, "table2_sub_index.pkl"), label)
    else:
        run_tasks(task, tasks, a.workers, os.path.join(a.out_dir, "table2_certificates.pkl"), label)
