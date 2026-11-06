"""Figure 4: policy ratios with the radius multiplier selected by repeated cross-validation.

Histories: (a) K = 5, beta in {0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7}, seeds 0-4;
           (b) beta = 0.25, K in {5, 10, ..., 40}, seeds 0-9.
For every history, c is selected from {0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 7, 10} separately for the CDR and the
subgradient policy (rwcmdp.validation), and the policies built from all transitions with the selected c are evaluated
under the true kernel. Writes two files to --out-dir:
  fig4_lp.pkl   records {K, seed, b, T, L, grid, cv_scores_cdr, c_cdr, 'F-MLE', 'random', 'F-CDR(cv)'}
  fig4_sub.pkl  records {kind: 'fig4', K, seed, b, grid, cv_scores, c_sub, ratio}
Full run: 125 histories (beta = 0 included), 30-45 min per history and policy, i.e. about a day with 6 workers.
Quick check (35-45 min per part):  python experiments/fig4_cross_validation.py --K 5 --beta 0.25 --seeds 0 --workers 2
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
from rwcmdp.validation import RADIUS_GRID, select_multiplier, validation_scores  # noqa: E402

BETAS_A = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7)
KS_B = (5, 10, 15, 20, 25, 30, 35, 40)


def default_histories():
    tasks = [(5, s, b) for b in BETAS_A for s in range(5)]
    tasks += [(K, s, 0.25) for K in KS_B for s in range(10) if (K, s, 0.25) not in tasks]
    return tasks


def lp_task(K, seed, beta, grid=RADIUS_GRID):
    """MLE and random policies, and the CDR policy with the validated multiplier."""
    h = history(K, seed)
    scores = validation_scores(h, beta, "cdr", grid)
    c = select_multiplier(scores, grid)
    v_true = ev.true_kernel_reward(h, beta)
    _, p_mle, _ = fluid_lp_policy(h.P_hat, h.R, h.q, beta, HORIZON)
    _, p_cdr = cdr_policy(h.P_hat, radius(c, K, h.L), h.R, h.q, beta, h.mask, HORIZON)
    return {"K": K, "seed": seed, "b": beta, "T": HORIZON, "L": h.L, "grid": tuple(grid), "cv_scores_cdr": scores,
            "c_cdr": c, "F-MLE": ev.reward(h, beta, p_mle) / v_true,
            "random": ev.reward(h, beta, random_policy(HORIZON, h.n_machines, h.P0.shape[1])) / v_true,
            "F-CDR(cv)": ev.reward(h, beta, p_cdr) / v_true}


def sub_task(K, seed, beta, grid=RADIUS_GRID):
    """Subgradient policy with the validated multiplier."""
    h = history(K, seed)
    scores = validation_scores(h, beta, "sub", grid)
    c = select_multiplier(scores, grid)
    v_true = ev.true_kernel_reward(h, beta)
    _, _, mu_mle = fluid_lp_policy(h.P_hat, h.R, h.q, beta, HORIZON)
    pol = subgradient_policy(h.P_hat, radius(c, K, h.L), h.R, h.q, beta, h.mask, HORIZON, mu_mle)
    return dict(kind="fig4", K=K, seed=seed, b=beta, grid=tuple(grid), cv_scores=scores, c_sub=c,
                ratio=ev.reward_index(h, beta, pol) / v_true)


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--K", type=int, nargs="+", default=None, help="restrict to these K")
    p.add_argument("--beta", type=float, nargs="+", default=None, help="restrict to these repair ratios")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="restrict to these histories")
    p.add_argument("--part", choices=["lp", "sub", "both"], default="both")
    a = p.parse_args()
    tasks = default_histories()
    if a.K is not None or a.beta is not None or a.seeds is not None:
        keep = lambda K, s, b: ((a.K is None or K in a.K) and (a.seeds is None or s in a.seeds)  # noqa: E731
                                and (a.beta is None or any(abs(b - x) < 1e-9 for x in a.beta)))
        tasks = [t for t in tasks if keep(*t)]
    tasks.sort(key=lambda t: (-t[0], t[2], t[1]))   # the long K = 40 histories first
    label = lambda t: f"K={t[0]} seed={t[1]} beta={t[2]:g}"  # noqa: E731
    if a.part in ("lp", "both"):
        run_tasks(lp_task, tasks, a.workers, os.path.join(a.out_dir, "fig4_lp.pkl"), label)
    if a.part in ("sub", "both"):
        run_tasks(sub_task, tasks, a.workers, os.path.join(a.out_dir, "fig4_sub.pkl"), label)
