"""Policies of Section 4.1 and the benchmarks of Section 5.

A separable policy is stored as the repair probability pi_ti(repair | s), an array of shape (T, I, S); the subgradient
policy is an index policy, stored as its index gains and tie-breaking probabilities (``IndexPolicy``). Feasibility of
the coupling constraint is restored in the simulation (``rwcmdp.simulation``).

* fluid policy pi^IF(x) of an occupancy measure x (Definition 1): pi_ti(a|s) = x_ti(s,a) / sum_a' x_ti(s,a'), and
  `do nothing` where the LP puts no mass on (t, i, s);
* MLE policy: fluid policy of FLP(p_hat); true-kernel policy: fluid policy of FLP(p0) (Definition 4);
* CDR policy: fluid policy of the CDR LP under the lower envelope (Section 3.3);
* subgradient policy (Definition 5): kernel sequence p~ of Algorithm 1, re-solve FLP(p~) for x~ and mu*(p~), index
  gains Q(s, repair) - Q(s, do nothing) under p~ and mu*(p~); ties broken by the fluid policy of x~;
* random policy: every machine requests a repair and beta * I requests are granted at random.
"""
from dataclasses import dataclass

import numpy as np

from . import lp
from .ambiguity import lower_envelope
from .relaxation import subgradient_method, worst_case_kernel_sequence

TIE_TOL = 1e-6   # index gains within TIE_TOL of zero count as ties


def fluid_policy(x, tol=1e-10):
    """Repair probabilities (T, I, S) of the fluid policy of x (T, I, S, A); `do nothing` where x has no mass."""
    mass = x.sum(-1)
    return np.where(mass > tol, np.clip(x[..., 1], 0, None) / np.maximum(mass, tol), 0.0)


def fluid_lp_policy(P, R, q, beta, horizon):
    """MLE policy (P = p_hat) or true-kernel policy (P = p0): returns (planning value, repair probabilities, mu*(P))."""
    value, x, mu = lp.solve_fluid_lp(P, R, q, beta, horizon)
    return value, fluid_policy(x), mu


def cdr_policy(P_hat, rho, R, q, beta, mask, horizon, flow_inequality=True):
    """CDR policy: returns (planning value L_CDR(P), repair probabilities)."""
    value, x, _ = lp.solve_cdr_lp(lower_envelope(P_hat, rho, mask), R, q, beta, horizon, flow_inequality=flow_inequality)
    return value, fluid_policy(x)


def random_policy(horizon, n_machines, n_states):
    """Every machine requests a repair in every state; the restoration grants beta * I requests at random."""
    return np.ones((horizon, n_machines, n_states))


def random_working_policy(horizon, n_machines, n_states, n_working=7):
    """Random policy that requests repairs only for working machines (states 1-7)."""
    prob = np.zeros((horizon, n_machines, n_states))
    prob[:, :, :n_working] = 1.0
    return prob


@dataclass
class IndexPolicy:
    """Index policy: index gains (T, I, S) and the repair probability used to break ties (T, I, S).

    A machine requests a repair if its index gain exceeds TIE_TOL, and with probability ``tie_prob`` if the gain is
    within TIE_TOL of zero. ``value`` is the optimal value of the fluid LP that defines the policy.
    """
    index: np.ndarray
    tie_prob: np.ndarray
    value: float
    mu: np.ndarray

    def request_probability(self):
        """Repair probabilities of the separable (unrestored) policy."""
        return np.where(self.index > TIE_TOL, 1.0, np.where(np.abs(self.index) <= TIE_TOL, self.tie_prob, 0.0))


def index_policy(P_seq, R, q, beta, horizon):
    """Optimal Lagrangian index policy of Brown and Smith (2020) for the kernel sequence P_seq (T-1, I, S, A, S').

    Solves FLP(P_seq) for x and mu*(P_seq), computes Q_t(s, a) = r(s, a) - mu*_t D(s, a) + sum_s' p_t(s'|s,a) W_{t+1}(s')
    backward in time and returns the index gains Q(s, repair) - Q(s, do nothing). Ties are broken with the fluid policy
    of x; where x has no mass, the tie-breaking probability is 1{index gain > 0}.
    """
    T = horizon
    value, x, d = lp.solve_fluid_lp(P_seq, R, q, beta, T)
    I, S = x.shape[1], x.shape[2]
    W = np.zeros((I, S))
    index = np.zeros((T, I, S))
    for t in reversed(range(T)):
        cont = np.einsum("isaj,ij->isa", P_seq[t], W) if t < T - 1 else np.zeros((I, S, 2))
        Q = R - d[t] * np.array([0.0, 1.0]) + cont
        index[t] = Q[..., 1] - Q[..., 0]
        W = Q.max(-1)
    mass = x.sum(-1)
    tie_prob = np.where(mass > 1e-10, np.clip(x[..., 1], 0, None) / np.maximum(mass, 1e-10), (index > 0).astype(float))
    return IndexPolicy(index, tie_prob, value, d)


def constant_sequence(P, horizon):
    """The time-invariant kernel P as a kernel sequence of shape (T-1, I, S, A, S')."""
    return np.broadcast_to(P, (horizon - 1,) + P.shape).copy()


def subgradient_policy(P_hat, rho, R, q, beta, mask, horizon, mu0, n_iter=150, eta=0.01):
    """Subgradient policy (Definition 5): index policy for the kernel sequence p~ returned by Algorithm 1.

    mu0 is the starting point of Algorithm 1 (the budget multipliers of FLP(p_hat) in the experiments). The returned
    policy's ``value`` is the planning value, the optimal value of FLP(p~).
    """
    _, mu, _ = subgradient_method(P_hat, rho, mu0, R, beta, mask, q, horizon, n_iter=n_iter, eta=eta)
    return index_policy(worst_case_kernel_sequence(P_hat, rho, mu, R, mask, horizon), R, q, beta, horizon)
