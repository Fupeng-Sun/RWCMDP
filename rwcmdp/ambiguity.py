"""l_inf-ball ambiguity sets around the MLE, intersected with the simplex over the known reachable states.

For every machine i and state-action pair (s, a), P_i(s, a) = {p in Simplex(S_i(s, a)) : ||p - p_hat_i(.|s,a)||_inf <= rho}.
The radius rho is either common to all pairs (rho = c / sqrt(K ell) with the radius multiplier c, Section 5) or given
per pair (the radii rho_{m,i}(s, a) of Lemma 1, which guarantee coverage of the true kernel with probability 1 - alpha).
A pair with a single reachable state has a singleton ambiguity set, whatever its radius.
"""
import numpy as np


def radius(c, K, L):
    """Common radius rho = c / sqrt(K ell) of the radius multiplier c."""
    return c / np.sqrt(K * L)


def worst_case_kernel(P_hat, rho, W_next, mask):
    """argmin_{p in P_i(s,a)} sum_s' p(s') W_i(s') for every (i, s, a): the inner minimization of the robust recursion.

    Greedy solution: start from the lower bounds max(p_hat - rho, 0) on the support and pour the remaining mass into
    the reachable states with the smallest W first, up to their upper bounds min(p_hat + rho, 1).
    P_hat has shape (I, S, A, S'), W_next shape (I, S'); rho is a scalar or broadcasts against (I, S, A, 1).
    """
    lower = np.maximum(P_hat - rho, 0.0) * mask
    capacity = np.minimum(P_hat + rho, 1.0) * mask - lower
    remaining = 1.0 - lower.sum(axis=-1, keepdims=True)
    order = np.broadcast_to(np.argsort(W_next, axis=-1)[:, None, None, :], P_hat.shape)
    cap_sorted = np.take_along_axis(capacity, order, axis=-1)
    filled_before = np.cumsum(cap_sorted, axis=-1) - cap_sorted
    add_sorted = np.clip(remaining - filled_before, 0.0, cap_sorted)
    add = np.empty_like(add_sorted)
    np.put_along_axis(add, order, add_sorted, axis=-1)
    return lower + add


def lower_envelope(P_hat, rho, mask):
    """Lower envelope p_low(s'|s,a) = min_{p in P_i(s,a)} p(s') of the ambiguity set (the kernel of the CDR problem).

    For the l_inf ball on the simplex: max{0, p_hat - rho, 1 - sum_{s'' != s'} min(p_hat(s'') + rho, 1)} on the
    support. Its rows sum to less than one when rho > 0 (the lower envelope is sub-stochastic).
    """
    upper = np.minimum(P_hat + rho, 1.0) * mask
    rest = upper.sum(axis=-1, keepdims=True) - upper
    return np.maximum.reduce([np.zeros_like(P_hat), P_hat - rho, 1.0 - rest]) * mask


def ambiguous_pairs(mask):
    """Boolean array (I, S, A): the state-action pairs with at least two reachable states."""
    return mask.sum(-1) >= 2


def valid_radius(N, mask, m, alpha=0.05):
    """Radii rho_{m,i}(s,a) = min{1, sqrt(log(2 M m / alpha) / (2 n_i(s,a)))} of Section 4.2 (Lemma 1).

    M counts the uncertain transition probabilities (i, (s,a), s') of the pairs with at least two reachable states;
    an unobserved pair receives rho = 1, i.e. the whole simplex over its reachable states.
    Returns the radii (shape (I, S, A, 1)) and M.
    """
    n = N.sum(-1, keepdims=True)
    ambiguous = (mask.sum(-1, keepdims=True) >= 2)
    M = int((mask * ambiguous).sum())
    rho = np.sqrt(np.log(2 * M * m / alpha) / (2 * np.maximum(n, 1)))
    return np.where(n > 0, np.minimum(rho, 1.0), 1.0), M


def covers(P_hat, P0, rho, mask):
    """Boolean array (I, S, A): whether the ambiguity set of each pair contains the true row p0(.|s,a)."""
    cover = (np.abs(P_hat - P0) <= rho + 1e-12) | (mask == 0)
    return cover.all(axis=-1)
