"""Monte Carlo evaluation of policies with feasibility restoration (Sections 3.3 and 5).

In every period, each machine draws a tentative action. If more than beta * I machines request a repair, requests
are reverted to `do nothing` until the budget holds:

* period-by-period restoration (the experiments): LP-based policies revert requests uniformly at random, the index
  (subgradient) policy reverts the requests with the smallest index gains; a reverted machine may request again later;
* sticky restoration (the analysis, Definition 3): in addition, a reverted machine takes `do nothing` in all later
  periods.

Every function returns the mean total reward over ``n_runs`` independent runs of the fleet. The random numbers are
drawn from ``np.random.default_rng(seed)`` in the same order for every policy, so that the same seed gives common
random numbers across policies.
"""
import numpy as np

from .data import sample_next_states
from .policies import TIE_TOL


def simulate(seed, repair_prob, P_true, R, q, capacity, n_runs):
    """Separable policy with repair probabilities (T, I, S) and random period-by-period restoration."""
    rng = np.random.default_rng(seed)
    T, n_machines = repair_prob.shape[:2]
    machine = np.arange(n_machines)
    state = sample_next_states(np.broadcast_to(q, (n_runs, n_machines, q.shape[-1])), rng.random((n_runs, n_machines)))
    total = np.zeros(n_runs)
    for t in range(T):
        wants = rng.random((n_runs, n_machines)) < repair_prob[t, machine, state]
        priority = np.where(wants, rng.random((n_runs, n_machines)), -1.0)
        rank = np.argsort(np.argsort(-priority, axis=1), axis=1)  # 0 = highest priority
        action = (wants & (rank < capacity)).astype(int)
        total += R[state, action].sum(axis=1)
        state = sample_next_states(P_true[machine, state, action], rng.random((n_runs, n_machines)))
    return total.mean()


def simulate_sticky(seed, repair_prob, P_true, R, q, capacity, n_runs):
    """Separable policy with random sticky restoration. Returns (mean total reward, mean number of reverted machines at T)."""
    rng = np.random.default_rng(seed)
    T, n_machines = repair_prob.shape[:2]
    S = P_true.shape[1]
    machine = np.arange(n_machines)
    state = sample_next_states(np.broadcast_to(q, (n_runs, n_machines, S)), rng.random((n_runs, n_machines)))
    frozen = np.zeros((n_runs, n_machines), dtype=bool)
    total = np.zeros(n_runs)
    n_frozen_t = []
    for t in range(T):
        wants = (rng.random((n_runs, n_machines)) < repair_prob[t, machine, state]) & ~frozen
        priority = np.where(wants, rng.random((n_runs, n_machines)), -1.0)
        rank = np.argsort(np.argsort(-priority, axis=1), axis=1)
        action = (wants & (rank < capacity)).astype(int)
        frozen |= wants & (action == 0)                       # reverted now -> do nothing from now on
        total += R[state, action].sum(axis=1)
        n_frozen_t.append(frozen.sum(1).mean())
        state = sample_next_states(P_true[machine, state, action], rng.random((n_runs, n_machines)))
    return total.mean(), n_frozen_t[-1]


def simulate_index(seed, policy, P_true, R, q, capacity, n_runs):
    """Index policy with period-by-period restoration: the capacity goes to the largest index gains, ties at random.

    A machine requests a repair if its index gain exceeds TIE_TOL, and with the tie-breaking probability if the gain
    is within TIE_TOL of zero.
    """
    index, tie_prob = policy.index, policy.tie_prob
    rng = np.random.default_rng(seed)
    T, n = index.shape[:2]
    machine = np.arange(n)
    state = sample_next_states(np.broadcast_to(q, (n_runs, n, q.shape[-1])), rng.random((n_runs, n)))
    total = np.zeros(n_runs)
    for t in range(T):
        idx = index[t, machine, state]
        u = rng.random((n_runs, n))
        wants = (idx > TIE_TOL) | ((np.abs(idx) <= TIE_TOL) & (u < tie_prob[t, machine, state]))
        priority = np.where(wants, np.where(idx > TIE_TOL, idx, 0.0) + 1e-9 * rng.random((n_runs, n)), -np.inf)
        rank = np.argsort(np.argsort(-priority, axis=1), axis=1)
        action = (wants & (rank < capacity)).astype(int)
        total += R[state, action].sum(axis=1)
        state = sample_next_states(P_true[machine, state, action], rng.random((n_runs, n)))
    return total.mean()


def simulate_index_sticky(seed, policy, P_true, R, q, capacity, n_runs):
    """Index policy with sticky restoration: reverted machines stay on `do nothing`; reverts by smallest index gain."""
    index, tie_prob = policy.index, policy.tie_prob
    rng = np.random.default_rng(seed)
    T, n = index.shape[:2]
    machine = np.arange(n)
    state = sample_next_states(np.broadcast_to(q, (n_runs, n, q.shape[-1])), rng.random((n_runs, n)))
    frozen = np.zeros((n_runs, n), dtype=bool)
    total = np.zeros(n_runs)
    for t in range(T):
        idx = index[t, machine, state]
        u = rng.random((n_runs, n))
        wants = ((idx > TIE_TOL) | ((np.abs(idx) <= TIE_TOL) & (u < tie_prob[t, machine, state]))) & ~frozen
        priority = np.where(wants, np.where(idx > TIE_TOL, idx, 0.0) + 1e-9 * rng.random((n_runs, n)), -np.inf)
        rank = np.argsort(np.argsort(-priority, axis=1), axis=1)
        granted = wants & (rank < capacity)
        frozen |= wants & ~granted
        action = granted.astype(int)
        total += R[state, action].sum(axis=1)
        state = sample_next_states(P_true[machine, state, action], rng.random((n_runs, n)))
    return total.mean()
