"""Section 5, valid radii: robust policies with the ambiguity sets of Section 4.2, which contain the true kernel with
probability at least 1 - alpha (alpha = 0.05; Hoeffding's inequality with a union bound over the M uncertain
transition probabilities and the m possible sample sizes; unobserved pairs receive the whole simplex).

Histories: beta in {0.5, 0.25}, K in {5, 10, 15, 20}, seeds 0-19 (the histories of Figure 3). Writes valid_radius.pkl
to --out-dir; records {K, seed, b, m, M, covered, rho_median, c_equiv, share_full, MLE, Sub, CDR}, where
covered = share of the ambiguous pairs whose set contains the true row, rho_median = median radius over the ambiguous
pairs, c_equiv = rho_median * sqrt(m), share_full = share of ambiguous pairs with radius 1 (the whole simplex), and
MLE / Sub / CDR are policy ratios. Full run: 160 tasks, about 10 min with 6 workers.
Quick check:  python experiments/valid_radius.py --K 5 --beta 0.5 --seeds 0
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._common import parser, run_tasks  # noqa: E402

import numpy as np  # noqa: E402

from rwcmdp import evaluation as ev  # noqa: E402
from rwcmdp.ambiguity import ambiguous_pairs, valid_radius  # noqa: E402
from rwcmdp.data import history  # noqa: E402
from rwcmdp.instance import HORIZON  # noqa: E402
from rwcmdp.policies import cdr_policy, fluid_lp_policy, subgradient_policy  # noqa: E402

ALPHA = 0.05


def task(K, seed, beta, alpha=ALPHA):
    h = history(K, seed)
    m = K * (h.L - 1)                                  # observed transitions per machine
    rho, M = valid_radius(h.N, h.mask, m, alpha)
    amb = ambiguous_pairs(h.mask)
    covered = (np.abs(h.P_hat - h.P0).max(-1) <= rho[..., 0] + 1e-12)[amb].mean()
    v_true = ev.true_kernel_reward(h, beta)
    _, p_mle, mu_mle = fluid_lp_policy(h.P_hat, h.R, h.q, beta, HORIZON)
    sub = subgradient_policy(h.P_hat, rho, h.R, h.q, beta, h.mask, HORIZON, mu_mle)
    _, p_cdr = cdr_policy(h.P_hat, rho, h.R, h.q, beta, h.mask, HORIZON)
    rho_amb = rho[..., 0][amb]
    return dict(K=K, seed=seed, b=beta, m=m, M=M, covered=covered, rho_median=float(np.median(rho_amb)),
                c_equiv=float(np.median(rho_amb) * np.sqrt(m)), share_full=float((rho_amb >= 1).mean()),
                MLE=ev.reward(h, beta, p_mle) / v_true, Sub=ev.reward_index(h, beta, sub) / v_true,
                CDR=ev.reward(h, beta, p_cdr) / v_true)


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--K", type=int, nargs="+", default=[5, 10, 15, 20])
    p.add_argument("--beta", type=float, nargs="+", default=[0.5, 0.25])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    a = p.parse_args()
    tasks = [(K, s, b) for b in a.beta for K in a.K for s in a.seeds]
    run_tasks(task, tasks, a.workers, os.path.join(a.out_dir, "valid_radius.pkl"),
              lambda t: f"K={t[0]} seed={t[1]} beta={t[2]:g}")
