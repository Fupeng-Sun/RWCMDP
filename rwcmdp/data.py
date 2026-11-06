"""Historical data and the maximum likelihood estimate (Sections 4.2 and 5).

For every machine, K trajectories of ell periods are generated under the true kernel by a behavior policy that repairs
with probability 0.3 in every state and period, which gives m = K (ell - 1) observed transitions per machine. The MLE
is n(s, a, s') / n(s, a) for an observed state-action pair and the uniform distribution over the known reachable
states otherwise.
"""
from dataclasses import dataclass

import numpy as np

from . import instance


def sample_next_states(P_rows, u):
    """Draw next states by inverse CDF; P_rows has shape (..., S), u has shape (...)."""
    cdf = np.cumsum(P_rows, axis=-1)
    return np.minimum((cdf < u[..., None]).sum(axis=-1), P_rows.shape[-1] - 1)


def generate_trajectories(rng, P_true, q, n_paths, path_length, repair_prob):
    """K trajectories of length ell per machine under the behavior policy A ~ Bernoulli(repair_prob).

    Returns states and actions, both of shape (I, K, ell).
    """
    n_machines = q.shape[0]
    states = np.zeros((n_machines, n_paths, path_length), dtype=int)
    actions = np.zeros((n_machines, n_paths, path_length), dtype=int)
    machine = np.arange(n_machines)[:, None]
    states[:, :, 0] = sample_next_states(q[:, None, :].repeat(n_paths, axis=1), rng.random((n_machines, n_paths)))
    for t in range(path_length):
        actions[:, :, t] = rng.random((n_machines, n_paths)) < repair_prob
        if t + 1 < path_length:
            rows = P_true[machine, states[:, :, t], actions[:, :, t]]
            states[:, :, t + 1] = sample_next_states(rows, rng.random((n_machines, n_paths)))
    return states, actions


def transition_counts(states, actions, n_states, mask=None, n_actions=instance.N_ACTIONS):
    """Counts n_i(s, a, s') of the observed transitions, shape (I, S, A, S').

    ``mask`` (shape (I, K, ell - 1)) selects a subset of the transitions; None uses all of them.
    """
    n_machines = states.shape[0]
    s, a, s_next = states[:, :, :-1], actions[:, :, :-1], states[:, :, 1:]
    machine = np.broadcast_to(np.arange(n_machines)[:, None, None], s.shape)
    keep = np.ones(s.shape, dtype=bool) if mask is None else mask
    N = np.zeros((n_machines, n_states, n_actions, n_states))
    np.add.at(N, (machine[keep], s[keep], a[keep], s_next[keep]), 1)
    return N


def mle_kernel(N, support):
    """MLE p_hat(s'|s,a) = n(s,a,s') / n(s,a); uniform over the reachable states of an unobserved pair."""
    N_sa = N.sum(-1, keepdims=True)
    prior = support / support.sum(-1, keepdims=True)
    return np.where(N_sa > 0, N / np.maximum(N_sa, 1), np.broadcast_to(prior, N.shape))


@dataclass
class History:
    """One data set of the experiments: the fleet, the observed trajectories and the MLE."""
    P0: np.ndarray        # true kernel (I, S, A, S')
    R: np.ndarray         # rewards (S, A)
    mask: np.ndarray      # known supports (I, S, A, S')
    q: np.ndarray         # initial distributions (I, S)
    states: np.ndarray    # observed states (I, K, ell)
    actions: np.ndarray   # observed actions (I, K, ell)
    N: np.ndarray         # transition counts (I, S, A, S')
    P_hat: np.ndarray     # MLE (I, S, A, S')
    K: int
    seed: int
    L: int                # trajectory length ell

    @property
    def n_machines(self):
        return self.q.shape[0]


def history(K, seed, path_length=instance.TRAJECTORY_LENGTH, behavior_repair_prob=instance.BEHAVIOR_REPAIR_PROB):
    """The history with K trajectories per machine for a given seed (random stream [7, seed])."""
    P0, R, mask, q = instance.fleet()
    rng = np.random.default_rng([7, seed])
    states, actions = generate_trajectories(rng, P0, q, K, path_length, behavior_repair_prob)
    N = transition_counts(states, actions, P0.shape[1])
    return History(P0, R, mask, q, states, actions, N, mle_kernel(N, mask), K, seed, path_length)
