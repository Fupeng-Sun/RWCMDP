"""Figure 3: policy ratios as a function of the radius multiplier c (beta = 0.5, K in {5, 10, 15, 20}, histories 0-19).

For every history (K, seed): the MLE and random policies, the CDR policy for every c in CDR_RADII and the subgradient
policy for every c in SUB_RADII, all evaluated under the true kernel and divided by the true-kernel policy's reward.
Writes two files to --out-dir:
  fig3_lp.pkl   records {K, seed, b, T, L, 'F-MLE', 'random', curves: {c: {'F-CDR': ratio}}}
  fig3_sub.pkl  records {kind: 'fig3', K, seed, b, curves: {c: ratio of the subgradient policy}}
Full run: 80 histories, about 2-3 h with 6 workers on an Apple M5 Pro (most of it in the CDR LPs).
Quick check:  python experiments/fig3_radius.py --K 5 --seeds 0 --radii 0 0.5 2 --workers 3
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._common import parser, run_tasks  # noqa: E402

from rwcmdp import evaluation as ev  # noqa: E402
from rwcmdp.ambiguity import radius  # noqa: E402
from rwcmdp.data import history  # noqa: E402
from rwcmdp.instance import HORIZON  # noqa: E402
from rwcmdp.policies import cdr_policy, fluid_lp_policy, random_policy, subgradient_policy  # noqa: E402

BETA = 0.5
CDR_RADII = (0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 10)
SUB_RADII = (0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5)


def lp_task(K, seed, radii, beta=BETA):
    """MLE, random and CDR policies of one history."""
    h = history(K, seed)
    v_true = ev.true_kernel_reward(h, beta)
    _, p_mle, _ = fluid_lp_policy(h.P_hat, h.R, h.q, beta, HORIZON)
    out = {"K": K, "seed": seed, "b": beta, "T": HORIZON, "L": h.L, "F-MLE": ev.reward(h, beta, p_mle) / v_true,
           "random": ev.reward(h, beta, random_policy(HORIZON, h.n_machines, h.P0.shape[1])) / v_true, "curves": {}}
    for c in radii:
        _, p_cdr = cdr_policy(h.P_hat, radius(c, K, h.L), h.R, h.q, beta, h.mask, HORIZON)
        out["curves"][c] = {"F-CDR": ev.reward(h, beta, p_cdr) / v_true}
    return out


def sub_task(K, seed, radii, beta=BETA):
    """Subgradient policy of one history; Algorithm 1 starts from the multipliers of FLP(p_hat)."""
    h = history(K, seed)
    v_true = ev.true_kernel_reward(h, beta)
    _, _, mu_mle = fluid_lp_policy(h.P_hat, h.R, h.q, beta, HORIZON)
    curves = {}
    for c in radii:
        pol = subgradient_policy(h.P_hat, radius(c, K, h.L), h.R, h.q, beta, h.mask, HORIZON, mu_mle)
        curves[c] = ev.reward_index(h, beta, pol) / v_true
    return dict(kind="fig3", K=K, seed=seed, b=beta, curves=curves)


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--K", type=int, nargs="+", default=[5, 10, 15, 20])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    p.add_argument("--radii", type=float, nargs="+", default=None, help="subset of radius multipliers (default: all)")
    p.add_argument("--part", choices=["lp", "sub", "both"], default="both")
    a = p.parse_args()
    cdr_radii = CDR_RADII if a.radii is None else tuple(a.radii)
    sub_radii = SUB_RADII if a.radii is None else tuple(a.radii)
    label = lambda t: f"K={t[0]} seed={t[1]}"  # noqa: E731
    if a.part in ("lp", "both"):
        run_tasks(lp_task, [(K, s, cdr_radii) for K in a.K for s in a.seeds], a.workers,
                  os.path.join(a.out_dir, "fig3_lp.pkl"), label)
    if a.part in ("sub", "both"):
        run_tasks(sub_task, [(K, s, sub_radii) for K in a.K for s in a.seeds], a.workers,
                  os.path.join(a.out_dir, "fig3_sub.pkl"), label)
