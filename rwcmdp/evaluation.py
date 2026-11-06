"""Evaluation protocol shared by all experiments of Section 5.

Random streams (all ``np.random.default_rng(list)``):
  [7, seed]             historical trajectories of the history ``seed`` (``rwcmdp.data.history``)
  [1, seed]             out-of-sample evaluation: 1,000 runs of the fleet under the true kernel, the same for every
                        policy, radius and K (common random numbers)
  [3, seed, K, fold]    random 60/40 split of a cross-validation fold
  [4, seed, K, fold]    300 simulation runs under the validation kernel of the fold

The policy ratio of a policy is its mean total reward under the true kernel divided by that of the true-kernel policy
(the fluid policy of FLP(p0) with random period-by-period restoration), both estimated with the stream [1, seed].
"""
from . import instance
from .policies import fluid_lp_policy
from .simulation import simulate, simulate_index

SIM_RUNS = 1000   # runs for the out-of-sample evaluation
CV_RUNS = 300     # runs for each validation score
SUBGRADIENT_ITERATIONS = 150


def eval_stream(seed):
    return [1, seed]


def capacity(beta, h):
    return instance.repair_capacity(beta, h.n_machines)


def true_kernel_reward(h, beta, horizon=instance.HORIZON):
    """Mean total reward of the true-kernel policy pi^TRUE under p0 (the denominator of every policy ratio)."""
    _, prob, _ = fluid_lp_policy(h.P0, h.R, h.q, beta, horizon)
    return simulate(eval_stream(h.seed), prob, h.P0, h.R, h.q, capacity(beta, h), SIM_RUNS)


def reward(h, beta, prob):
    """Mean total reward under p0 of a separable policy (repair probabilities) with random restoration."""
    return simulate(eval_stream(h.seed), prob, h.P0, h.R, h.q, capacity(beta, h), SIM_RUNS)


def reward_index(h, beta, policy):
    """Mean total reward under p0 of an index policy with index-priority restoration."""
    return simulate_index(eval_stream(h.seed), policy, h.P0, h.R, h.q, capacity(beta, h), SIM_RUNS)
